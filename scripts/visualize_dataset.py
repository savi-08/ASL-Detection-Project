import os
import cv2
import matplotlib.pyplot as plt

# Path to training data
data_path = r"C:\Users\shrav\Desktop\ASL_Detection_Project\data\asl_alphabet_train"

# List class folders (A-Z, SPACE, DELETE, NOTHING)
classes = sorted(os.listdir(data_path))
print(f"Total classes found: {len(classes)}")
print("Classes:", classes)

# Display one sample image per class (up to 20)
plt.figure(figsize=(15, 10))

for idx, class_name in enumerate(classes[:20]):
    class_dir = os.path.join(data_path, class_name)
    image_name = os.listdir(class_dir)[0]
    image_path = os.path.join(class_dir, image_name)

    # Read and convert image from BGR to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(4, 5, idx + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()
