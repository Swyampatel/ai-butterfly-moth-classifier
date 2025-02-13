from app import predict_from_path

image_path = "C:/Users/rudra/Downloads/sample_butterfly.jpg"  # Change this to your image path
result = predict_from_path(image_path)

print(f"Species: {result['species']}")
print(f"Confidence: {result['confidence']:.2f}")
