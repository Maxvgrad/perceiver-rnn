import random
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageEnhance

class AddShadow:
    def __init__(self, shadow_count=1, shadow_dimension=(0.4, 0.6), intensity=0.5):
        """
        Initializes the AddRaindrops transformation.
        drop_count_range: Tuple indicating the minimum and maximum number of raindrops to add.
        drop_size_range: Tuple indicating the minimum and maximum size of the raindrops.
        drop_transparency_range: Tuple indicating the minimum and maximum transparency of raindrops.
        """
        self.shadow_count = shadow_count
        self.shadow_dimension = shadow_dimension
        self.intensity = intensity

    def __call__(self, image):
        image = np.array(image)

        h, w = image.shape[0], image.shape[1]
        for _ in range(self.shadow_count):
            top_x, bottom_x = random.randint(0, w), random.randint(0, w)
            top_y, bottom_y = random.randint(0, h // 2), random.randint(h // 2, h)
            shadow_height = int(random.uniform(*self.shadow_dimension) * h)

            polygon = np.array([
                [top_x, top_y],
                [bottom_x, bottom_y],
                [bottom_x, bottom_y + shadow_height],
                [top_x, top_y + shadow_height]
            ])

            overlay = image.copy()
            cv2.fillPoly(overlay, [polygon], (0, 0, 0))

            # Blend the shadow with the original image
            image = cv2.addWeighted(overlay, self.intensity, image, 1 - self.intensity, 0)

        return Image.fromarray(image)

class AddSnowdrops:
    def __init__(self, drop_count_range=(30, 100), drop_size_range=(2, 10), drop_transparency_range=(0.3, 0.8)):
        """
        Initializes the AddSnowdrops transformation.
        drop_count_range: Tuple indicating the minimum and maximum number of raindrops.
        drop_size_range: Tuple indicating the minimum and maximum size of the raindrops.
        drop_transparency_range: Tuple indicating the minimum and maximum transparency of raindrops.
        """
        self.drop_count_range = drop_count_range
        self.drop_size_range = drop_size_range
        self.drop_transparency_range = drop_transparency_range

    def __call__(self, img):
        img = np.array(img)
        h, w, _ = img.shape
        
        overlay = img.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        
        drop_count = random.randint(*self.drop_count_range) 
        
        for _ in range(drop_count):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            drop_size = random.randint(*self.drop_size_range)
            transparency = random.uniform(*self.drop_transparency_range)
            
            cv2.circle(overlay, (x, y), drop_size, (255, 255, 255), -1)
            cv2.circle(mask, (x, y), drop_size, 255, -1)
        
        out = cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img, cv2.CV_8UC3)
        img = np.where(mask[..., None].astype(bool), out, img)

        return Image.fromarray(img)
    
class AddRainStreaks:
    def __init__(self, streak_count=50, streak_length_range=(10, 20), streak_angle_range=(85, 95), transparency_range=(0.2, 0.5)):
        """
        Initializes the AddRainStreaks transformation.
        streak_count: Number of rain streaks to add.
        streak_length_range: Tuple indicating the minimum and maximum length of the rain streaks.
        streak_angle_range: Tuple indicating the minimum and maximum angle deviation from vertical (90 degrees).
        transparency_range: Tuple indicating the minimum and maximum transparency of the rain streaks.
        """
        self.streak_count = streak_count
        self.streak_length_range = streak_length_range
        self.streak_angle_range = streak_angle_range
        self.transparency_range = transparency_range

    def __call__(self, img):
        img = np.array(img)
        h, w, _ = img.shape
        
        overlay = img.copy()
        
        for _ in range(self.streak_count):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            streak_length = random.randint(*self.streak_length_range)
            angle = random.uniform(*self.streak_angle_range)
            transparency = random.uniform(*self.transparency_range)
            
            # Calculate end point of the streak
            end_x = int(x + streak_length * np.cos(np.radians(angle)))
            end_y = int(y + streak_length * np.sin(np.radians(angle)))
            
            cv2.line(overlay, (x, y), (end_x, end_y), (255, 255, 255), 1)
        
        # Blend the original image with the overlay
        img = cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)

        return Image.fromarray(img)    
    
class ImageTransform:
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomChoice([
                AddShadow(shadow_count=2, shadow_dimension=(0.2, 0.4), intensity=0.5),
                AddSnowdrops(drop_count_range=(10, 200), drop_size_range=(0, 2), drop_transparency_range=(0.2, 0.5)),
                AddRainStreaks(streak_count=200, streak_length_range=(3, 12), streak_angle_range=(75, 115), transparency_range=(0.15, 0.40)),
                transforms.GaussianBlur(kernel_size=5),
                transforms.RandomAdjustSharpness(sharpness_factor=3),
            ]),
            transforms.ToTensor(),  # Ensure this is the final step
        ])

    def __call__(self, data):
        image = data['image']
        image = TF.to_pil_image(image)  # Ensure image is in PIL format
        image = self.augmentations(image)
        data['image'] = image
        return data
    
class Normalize(object):
    def __call__(self, data, transform=None):
        image = data["image"]
        image = image / 255
        data["image"] = image
        return data