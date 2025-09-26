import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 🏷️ ชื่อแอป
st.title("🛡️ YOLO Helmet Detection App")

# 🚀 โหลดโมเดล YOLOv8
model = YOLO("best.pt")  # ปรับ path ตามโมเดลของคุณ

# 📤 อัปโหลดภาพ
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

# ✅ ถ้ามีภาพที่อัปโหลด
if uploaded_image is not None:
    # 🖼️ แสดงภาพต้นฉบับ
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # 🔄 แปลงภาพเป็น numpy array
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # 🔍 รัน YOLOv8 ตรวจจับวัตถุ
    st.info("Running YOLO object detection...")
    results = model.predict(image_np, conf=0.4)

    # 🖌️ วาดผลลัพธ์บนภาพ
    result_image = results[0].plot()
    st.image(result_image, caption="Detection Result", use_container_width=True)
    st.success("Detection completed!")

    # 📦 ดึงผลลัพธ์การตรวจจับ
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]

    # 🪖 นับจำนวนหมวกนิรภัย
    helmet_count = class_names.count("Helmet")  # ตรวจสอบว่า "Helmet" ตรงกับ model.names
    st.write(f"🪖 Number of Helmet detected: **{helmet_count}**")
