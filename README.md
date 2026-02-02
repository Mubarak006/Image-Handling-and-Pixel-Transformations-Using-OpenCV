# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** Mubarak R 
- **Register Number:** 212224220066

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

```

#### 2. Print the image width, height & Channel.
```python
img.shape
```
Output:
```
(600, 768, 3)
```

#### 3. Display the image using matplotlib imshow().
```python
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray,cmap='gray')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
img=cv2.imread('Eagle_in_Flight.jpg')
cv2.imwrite('Eagle_in_Flight.jpg',img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img=cv2.imread('Eagle_in_Flight.jpg')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(img)
plt.show()
img.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
crop = img_rgb[0:450,200:550] 
plt.imshow(crop[:,:,::-1])
plt.title("Cropped Region")
plt.axis("off")
plt.show()
crop.shape
```

#### 8. Resize the image up by a factor of 2x.
```python
res= cv2.resize(crop,(200*2, 200*2))
```

#### 9. Flip the cropped/resized image horizontally.
```python
flip= cv2.flip(res,1)
plt.imshow(flip[:,:,::-1])
plt.title("Flipped Horizontally")
plt.axis("off")
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img=cv2.imread('Eagle_in_Flight.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb.shape
```
Output:
```
(600, 768, 3)
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = cv2.putText(img_rgb, "Apollo 11 Saturn V Launch, July 16, 1969", (300, 700),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
plt.imshow(text, cmap='gray')  
plt.title("New image")
plt.show()
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rcol= (255, 0, 255)
cv2.rectangle(img_rgb, (400, 100), (800, 650), rcol, 3)
```
Output:
```
array([[[120, 151, 180],
        [119, 150, 179],
        [122, 153, 182],
        ...,
        [ 76, 107, 136],
        [ 78, 107, 139],
        [ 80, 109, 141]],

       [[123, 154, 183],
        [121, 152, 181],
        [120, 151, 180],
        ...,
        [ 75, 106, 135],
        [ 74, 105, 136],
        [ 73, 104, 135]],

       [[121, 152, 181],
        [121, 152, 181],
        [119, 150, 179],
        ...,
        [ 71, 105, 133],
        [ 69, 103, 131],
        [ 67, 101, 129]],

       ...,

       [[ 96, 125, 155],
        [ 99, 128, 158],
        [ 99, 128, 158],
        ...,
        [106, 133, 162],
        [110, 137, 166],
        [111, 138, 167]],

       [[106, 133, 163],
        [112, 139, 169],
        [113, 139, 172],
        ...,
        [106, 133, 163],
        [102, 131, 161],
        [ 96, 125, 155]],

       [[103, 130, 160],
        [111, 138, 168],
        [114, 140, 173],
        ...,
        [ 98, 125, 155],
        [ 92, 121, 151],
        [ 86, 115, 145]]], shape=(600, 768, 3), dtype=uint8)
```

#### 13. Display the final annotated image.
```python
plt.title("Annotated image")
plt.imshow(img_rgb)
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
img =cv2.imread('Eagle_in_Flight.jpg',cv2.IMREAD_COLOR)
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
```python
m = np.ones(img_rgb.shape, dtype="uint8") * 50
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img_rgb, m)  
img_darker = cv2.subtract(img_rgb, m)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_brighter), plt.title("Brighter Image"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_darker), plt.title("Darker Image"), plt.axis("off")
plt.show()
```

#### 18. Modify the image contrast.
```python
matrix1 = np.ones(img_rgb.shape, dtype="float32") * 1.1
matrix2 = np.ones(img_rgb.shape, dtype="float32") * 1.2
img_higher1 = cv2.multiply(img.astype("float32"), matrix1).clip(0,255).astype("uint8")
img_higher2 = cv2.multiply(img.astype("float32"), matrix2).clip(0,255).astype("uint8")
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_higher1), plt.title("Higher Contrast (1.1x)"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_higher2), plt.title("Higher Contrast (1.2x)"), plt.axis("off")
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(b, cmap='gray'), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(g, cmap='gray'), plt.title("Green Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(r, cmap='gray'), plt.title("Red Channel"), plt.axis("off")
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merged_rgb = cv2.merge([r, g, b])
plt.figure(figsize=(5,5))
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis("off")
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(h, cmap='gray'), plt.title("Hue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(s, cmap='gray'), plt.title("Saturation Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(v, cmap='gray'), plt.title("Value Channel"), plt.axis("off")
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
combined = np.concatenate((img_rgb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image  &  Merged HSV Image")
plt.axis("off")
plt.show()
```

## Output:

  1..Read 'Eagle_in_Flight.jpg' as grayscale and display:
  <img width="616" height="473" alt="Screenshot 2026-02-02 121036" src="https://github.com/user-attachments/assets/afc8befc-4657-4539-a3bc-16c02c3e3aa0" />

  2.Save image as PNG and display:
 <img width="629" height="456" alt="image" src="https://github.com/user-attachments/assets/a38b0dd2-fcc0-4d8d-803a-6bf154ee9b1c" />

  3. Cropped Region"
  <img width="389" height="480" alt="image" src="https://github.com/user-attachments/assets/fed67ac3-2811-4f66-a6de-a7ab44f3c439" />


  4. Flipped Image:
  <img width="481" height="484" alt="image" src="https://github.com/user-attachments/assets/007d316f-5811-4ff6-bfb4-5c03dde8f2d7" />

  5. Annotated Image:
  <img width="649" height="521" alt="image" src="https://github.com/user-attachments/assets/16494b63-46ee-44c2-9547-2c888854be7b" />

  6.Display the images (Original Image, Darker Image, Brighter Image):
  <img width="978" height="294" alt="image" src="https://github.com/user-attachments/assets/4e5c06bd-c733-4642-b74e-ab38c1582b7f" />

  7.Display the images (Original, Lower Contrast, Higher Contrast):
  <img width="973" height="268" alt="image" src="https://github.com/user-attachments/assets/5b58cd0e-f9df-4f72-ae7d-c1e99a10a1f7" />


  8.Split the image into the B,G,R components & Display the channels:
  
  <img width="924" height="260" alt="image" src="https://github.com/user-attachments/assets/73a7f533-40b6-4add-9e4d-c12611c5b1ab" />

  9.merged the R, G, B , display along with orginal image:
  
  <img width="485" height="397" alt="image" src="https://github.com/user-attachments/assets/bdd70ed4-4d81-4eda-8a81-2f11a1fe1b2c" />

  10.Split the image into the H, S, V components & Display the channels:
  
  <img width="936" height="285" alt="image" src="https://github.com/user-attachments/assets/22c12dbc-a6c9-4634-82cc-a6f816ee9629" />

  11.merged the H, S, V, display along with orginal image:
  <img width="895" height="390" alt="image" src="https://github.com/user-attachments/assets/8e8d2e20-1a19-4da5-8cbd-a34274a006b8" />



## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

