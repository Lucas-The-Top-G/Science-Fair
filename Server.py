from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from play import get_response
import re

app = FastAPI()

# Allow all origins (adjust as needed for your app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins instead of "*" for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_words():
    sentence = "This is a test response from the server. And there is not any illegal homes here."
    words = sentence.split()  # Split into words
    for word in words:
        yield word + " "  # Send one word at a time
        time.sleep(0.1)  # Simulate processing delay

@app.get("/process")
async def stream_response(
                Exterior: str = Header(...), 
                Parking: str = Header(...),
                Heating: str = Header(...),
                Sub: str = Header(...),
                Price: str = Header(...),
                SQFT: str = Header(...),
                Acres: str = Header(...),
                Beds: str = Header(...),
                Baths: str = Header(...),
                Description: str = Header(...),
            ):

    exterior_data = get_arr_to_string(Exterior)
    parking_data = get_arr_to_string(Parking)
    heating_data = get_arr_to_string(Heating)
    Price, SQFT, Acres, Beds, Baths = int(Price), int(SQFT), int(Acres), int(Beds), int(Baths)

    res =  get_response(Price, SQFT, Beds, Baths, Acres, exterior_data, heating_data, Sub, parking_data, Description)

    return {"": res}

def get_arr_to_string(s: str, des:bool=False):
    s = s.lower()
    s = s.replace(".", " thisistheendofthesentenceword ")
    s = s.replace(",", " acommashouldgohere")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip() 
    return s