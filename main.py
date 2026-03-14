from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import httpx
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="EatSpot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class SearchRequest(BaseModel):
    prompt: str
    latitude: float
    longitude: float
    radius: int = 3000

PARSE_PROMPT = """You are a food search assistant for India. Extract search parameters from the user's food query.
Return ONLY a valid JSON object, no extra text:
{
  "cuisine": "string or null",
  "budget_label": "budget | mid-range | premium | any",
  "budget_max_inr": "number or null",
  "open_after": "string or null",
  "dietary": "vegetarian | non-vegetarian | vegan | jain | halal | any",
  "vibe": "string or null",
  "search_query": "short food search term like biryani or vada pav or cafe",
  "out_of_area_ok": false
}"""

RANK_PROMPT = """You are a food recommendation expert for India.
Given a user's food request and a list of places, re-rank them by how well they match.
Return ONLY a valid JSON array, no extra text:
[{"place_id": "...", "match_score": 85, "match_reason": "One line explaining why this matches"}]
Order by match_score highest first. Include every place. Score 0-100."""

@app.get("/")
async def root():
    return {"message": "EatSpot API is live!", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/search")
async def search(req: SearchRequest):
    # STEP 1 - Parse prompt with Groq
    try:
        parse_res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[
                {"role": "system", "content": PARSE_PROMPT},
                {"role": "user", "content": req.prompt}
            ]
        )
        parsed = json.loads(parse_res.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt parsing failed: {str(e)}")

    search_query = parsed.get("search_query", req.prompt)

    # STEP 2 - Search OpenStreetMap via Overpass API (completely free)
    places = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Search for food places using Overpass API
            overpass_query = f"""
            [out:json][timeout:10];
            (
              node["amenity"~"restaurant|cafe|fast_food|food_court|bar|pub"]["name"~"{search_query}",i](around:{req.radius},{req.latitude},{req.longitude});
              node["amenity"~"restaurant|cafe|fast_food|food_court"]["cuisine"~"{search_query}",i](around:{req.radius},{req.latitude},{req.longitude});
              node["amenity"~"restaurant|cafe|fast_food|food_court"]["name"](around:{req.radius},{req.latitude},{req.longitude});
            );
            out body 20;
            """
            r = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": overpass_query}
            )
            data = r.json()
            osm_places = data.get("elements", [])

            for p in osm_places[:15]:
                tags = p.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                lat = p.get("lat", req.latitude)
                lon = p.get("lon", req.longitude)
                places.append({
                    "place_id": str(p.get("id")),
                    "name": name,
                    "rating": None,
                    "price_level": None,
                    "address": tags.get("addr:street", tags.get("addr:full", "Nearby")),
                    "open_now": None,
                    "photo_url": None,
                    "location": {"lat": lat, "lng": lon},
                    "types": [tags.get("amenity", "restaurant")],
                    "review_count": 0,
                    "cuisine": tags.get("cuisine", ""),
                    "phone": tags.get("phone", ""),
                    "website": tags.get("website", ""),
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Map search failed: {str(e)}")

    if not places:
        # Fallback - search by amenity type only if name search returns nothing
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                overpass_query = f"""
                [out:json][timeout:10];
                node["amenity"~"restaurant|cafe|fast_food"](around:{req.radius},{req.latitude},{req.longitude});
                out body 15;
                """
                r = await client.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": overpass_query}
                )
                data = r.json()
                for p in data.get("elements", [])[:15]:
                    tags = p.get("tags", {})
                    name = tags.get("name")
                    if not name:
                        continue
                    places.append({
                        "place_id": str(p.get("id")),
                        "name": name,
                        "rating": None,
                        "price_level": None,
                        "address": tags.get("addr:street", tags.get("addr:full", "Nearby")),
                        "open_now": None,
                        "photo_url": None,
                        "location": {"lat": p.get("lat"), "lng": p.get("lon")},
                        "types": [tags.get("amenity", "restaurant")],
                        "review_count": 0,
                        "cuisine": tags.get("cuisine", ""),
                    })
        except Exception:
            pass

    if not places:
        return {"places": [], "parsed": parsed, "total": 0}

    # STEP 3 - Re-rank with Groq
    summary = [
        {
            "place_id": p["place_id"],
            "name": p["name"],
            "address": p["address"],
            "types": p["types"],
            "cuisine": p.get("cuisine", ""),
        }
        for p in places
    ]

    try:
        rank_res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            messages=[
                {"role": "system", "content": RANK_PROMPT},
                {"role": "user", "content": f"User asked: \"{req.prompt}\"\n\nPlaces:\n{json.dumps(summary, indent=2)}"}
            ]
        )
        content = rank_res.choices[0].message.content.strip()
        # Extract JSON array from response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            rankings = json.loads(content[start:end])
            rank_map = {r["place_id"]: r for r in rankings}
        else:
            rank_map = {}
    except Exception:
        rank_map = {}

    # Merge rankings with place data
    for p in places:
        pid = p["place_id"]
        rank = rank_map.get(pid, {"match_score": 60, "match_reason": "Nearby food option"})
        p["match_score"] = rank.get("match_score", 60)
        p["match_reason"] = rank.get("match_reason", "Nearby food option")

    places.sort(key=lambda x: x["match_score"], reverse=True)
    return {"places": places, "parsed": parsed, "total": len(places)}


@app.get("/places/{place_id}")
async def place_detail(place_id: str):
    # Get details from OpenStreetMap
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://nominatim.openstreetmap.org/lookup",
                params={
                    "osm_ids": f"N{place_id}",
                    "format": "json",
                    "addressdetails": 1,
                    "extratags": 1,
                },
                headers={"User-Agent": "EatSpot/1.0"}
            )
            data = r.json()
    except Exception:
        raise HTTPException(status_code=404, detail="Place not found")

    if not data:
        raise HTTPException(status_code=404, detail="Place not found")

    p = data[0] if isinstance(data, list) else data
    tags = p.get("extratags", {})
    address = p.get("address", {})

    addr_parts = [
        address.get("road", ""),
        address.get("suburb", ""),
        address.get("city", ""),
    ]
    full_address = ", ".join(part for part in addr_parts if part)

    return {
        "place_id": place_id,
        "name": p.get("display_name", "").split(",")[0],
        "rating": None,
        "review_count": 0,
        "address": full_address or p.get("display_name", ""),
        "phone": tags.get("phone", None),
        "website": tags.get("website", None),
        "price_level": None,
        "opening_hours": [tags.get("opening_hours", "")] if tags.get("opening_hours") else [],
        "open_now": None,
        "photos": [],
        "reviews": [],
        "location": {
            "lat": float(p.get("lat", 0)),
            "lng": float(p.get("lon", 0))
        },
        "types": ["restaurant"],
    }


@app.get("/trending")
async def trending(lat: float = 19.0760, lng: float = 72.8777):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            overpass_query = f"""
            [out:json][timeout:10];
            node["amenity"~"restaurant|cafe|fast_food"]["name"](around:2000,{lat},{lng});
            out body 8;
            """
            r = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": overpass_query}
            )
            data = r.json()
            places = []
            for p in data.get("elements", [])[:8]:
                tags = p.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                places.append({
                    "place_id": str(p.get("id")),
                    "name": name,
                    "rating": None,
                    "address": tags.get("addr:street", "Nearby"),
                    "photo_url": None,
                    "price_level": None,
                })
            return {"trending": places}
    except Exception:
        return {"trending": []}
