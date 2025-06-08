import torch
import torchio as tio
import matplotlib.pyplot as plt

# 建立中心十字圖像
img = torch.zeros(1, 1, 32, 32, 32)
img[:, :, 16, :, :] = 1
img[:, :, :, 16, :] = 1
img[:, :, :, :, 16] = 1

scalar_img = tio.ScalarImage(tensor=img)

# 隨機旋轉（90 度）
transform = tio.RandomAffine(degrees=(0, 90, 0), p=1.0)
rotated = transform(scalar_img)

# 儲存 2D 切片圖像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img[0, 0, :, :, 16], cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(rotated.data[0, 0, :, :, 16], cmap='gray')
plt.title('Rotated')

plt.tight_layout()
plt.savefig("rotate_comparison.png")
print("✅ 圖片已儲存為 rotate_comparison.png")
