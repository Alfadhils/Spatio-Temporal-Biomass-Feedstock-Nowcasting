import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
from sklearn.preprocessing import OrdinalEncoder

class data_prep:
    def __init__(self, dfpath='dataset\Biomass_History.csv', imagepath='bio_images', image=True):
        self.dfbio = pd.read_csv(dfpath)
        self.folder_path = imagepath
        self.encoder = OrdinalEncoder()

        if image : 
            latlonpair = pd.DataFrame(self.encoder.fit_transform(self.dfbio[['Latitude','Longitude']].values), columns=['lat_encode','long_encode'])
            latlonpair['lat_encode'] = abs(latlonpair['lat_encode'] - latlonpair['lat_encode'].max())
            self.dfbio[['Latitude','Longitude']] = latlonpair
            self.selected_pix = self.dfbio[['Longitude','Latitude']].astype(int).values
        
    def create_images(self, width=75, height=60):
        self.bio_images = []
        for i,year in enumerate(range(2010,2018)):
            image = Image.new("F", (width,height))
            draw = ImageDraw.Draw(image)
            for index, row in self.dfbio.iterrows():
                x = int(row['Longitude'])
                y = int(row['Latitude'])
                value = row[str(year)]

                draw.point((x, y), fill=(value))
            
            self.bio_images.append(np.array(image))
            
        self.bio_images = np.array(self.bio_images)
    
    def save_images(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        min_value = np.min(self.bio_images)
        max_value = np.max(self.bio_images)

        for i, img in enumerate(self.bio_images):
            scaled_img = 255 * (img - min_value) / (max_value - min_value)
            scaled_img = np.clip(scaled_img, 0, 255).astype('uint8')

            image = Image.fromarray(scaled_img)
            image.save(os.path.join(self.folder_path, f'image_{2010+i}.png'))

    def get_images(self):
        return np.array(self.bio_images)

dp = data_prep()
dp.create_images(80,64)
dp.save_images()