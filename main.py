from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np


# สร้าง FastAPI App
app = FastAPI()

# โมเดลข้อมูลที่รับจาก Frontend
class PredictData(BaseModel):
    province: str
    district: str
    rubber_area: float
    rubber_tree_count: int
    rubber_type: str
    rubber_tree_age: int
    fer_top: str
    soil_group: str
    pH_top: str
    pH_low: str

# โหลดโมเดลและตัวแปลง (Encoders)
model_path = r'D:\DataSet_Project\seed-ui-devlopment\seed-back-devlopment\model\rubber_model.pkl'
encoders_path = r'D:\DataSet_Project\seed-ui-devlopment\seed-back-devlopment\model\encoders2.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoders_path, 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

# ฟังก์ชันสำหรับพยากรณ์
def predict_rubber_content(data: PredictData):
    try:
        # ดึง encoders จากไฟล์ที่โหลด
        encoder_province = encoders['encoder_province']
        encoder_district = encoders['encoder_district']
        encoder_rubbertype = encoders['encoder_rubbertype']
        encoder_fer_top = encoders['encoder_fer_top']
        encoder_soilgroup = encoders['encoder_soilgroup']
        encoder_pH_top = encoders['encoder_pH_top']
        encoder_pH_low = encoders['encoder_pH_low']

        # แปลงข้อมูลด้วย One-Hot Encoding และแปลงเป็น int หรือ bool
        new_province_encoded = encoder_province.transform(pd.DataFrame({'province': [data.province]})).astype(int)
        new_district_encoded = encoder_district.transform(pd.DataFrame({'district': [data.district]})).astype(int)
        new_rubber_type_encoded = encoder_rubbertype.transform(pd.DataFrame({'rubbertype': [data.rubber_type]})).astype(int)
        new_fer_top_encoded = encoder_fer_top.transform(pd.DataFrame({'fer_top': [data.fer_top]})).astype(int)
        new_soilgroup_encoded = encoder_soilgroup.transform(pd.DataFrame({'soilgroup': [data.soil_group]})).astype(int)
        new_pH_top_encoded = encoder_pH_top.transform(pd.DataFrame({'pH_top': [data.pH_top]})).astype(int)
        new_pH_low_encoded = encoder_pH_low.transform(pd.DataFrame({'pH_low': [data.pH_low]})).astype(int)

        # รวมข้อมูลทั้งหมดเข้าด้วยกัน
        input_data = pd.concat([
            pd.DataFrame(new_province_encoded, columns=encoder_province.get_feature_names_out(['province'])),
            pd.DataFrame(new_district_encoded, columns=encoder_district.get_feature_names_out(['district'])),
            pd.DataFrame(new_rubber_type_encoded, columns=encoder_rubbertype.get_feature_names_out(['rubbertype'])),
            pd.DataFrame(new_fer_top_encoded, columns=encoder_fer_top.get_feature_names_out(['fer_top'])),
            pd.DataFrame(new_soilgroup_encoded, columns=encoder_soilgroup.get_feature_names_out(['soilgroup'])),
            pd.DataFrame(new_pH_top_encoded, columns=encoder_pH_top.get_feature_names_out(['pH_top'])),
            pd.DataFrame(new_pH_low_encoded, columns=encoder_pH_low.get_feature_names_out(['pH_low']))
        ], axis=1)

        # เพิ่มข้อมูลที่ไม่ผ่านการแปลง (เช่น rubber_area, rubber_tree_count, etc.)
        input_data['rubberareaMeters'] = data.rubber_area
        input_data['rubbertreecount'] = data.rubber_tree_count
        input_data['rubbertreeage'] = data.rubber_tree_age

        # ทำนายผล
        predicted_yield = float(model.predict(input_data)[0])

        # เตรียมผลลัพธ์
        prediction_result = {
            "recommendation": f"ผลผลิตที่คาดว่าจะได้: {predicted_yield:.2f} กิโลกรัม",
            "predicted_yield": predicted_yield
        }

        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# Endpoint สำหรับรับข้อมูลและพยากรณ์
@app.post("/api/predict")
async def predict_endpoint(data: PredictData):
    try:
        result = predict_rubber_content(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
