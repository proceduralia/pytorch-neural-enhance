"""
Scrape the MIT/Adobe dataset website for downloading the images.
Save semantic info and paths in csv file 'mitdatainfo.csv'.

Create a directory hierarchy of the type:
base_dir/
	or_dir/
		0.png
		...
	expert_dir[0]/
		0.png
		...
	expert_dir[1]/
		0.png
		...
	...
"""
import requests
from lxml import html
import os
from PIL import Image
import pandas as pd
import rawpy
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_link', default='https://data.csail.mit.edu/graphics/fivek', help="Base link of website")
parser.add_argument('--base_dir', default='fivek', help="Path of the base directory for the dataset")
parser.add_argument('--size', default=500, help="Size for the greatest dimension of images")
parser.add_argument('--limit_number', default=0, type=int, help="Limit the number of images to scrape. 0 -> no limit")
parser.add_argument('--reverse', action='store_true',  help="Use reverse order for downloading the images")
args = vars(parser.parse_args())

#Parameters
base_link = args['base_link']
base_dir = args['base_dir']
or_dir = os.path.join(base_dir, 'original')
expert_dirs = [os.path.join(base_dir, 'expert{}'.format(i)) for i in range(5)]
size = args['size'] #maximum size for a dimension
limit_number = args['limit_number']
reverse = args['reverse']

page = requests.get(base_link)
tree = html.fromstring(page.content)
numbers = tree.xpath("//table[@class='data']//tr/td[1]/text()")
original_links = tree.xpath("//table[@class='data']//tr/td[3]/a/@href")
#Create list of lists (number_of_ims, number_of_experts)
expert_links = list(map(list, zip(*[tree.xpath("//table[@class='data']//tr/td[{}]/a/@href".format(i))
                                    for i in range(4, 9)])))
subjects = [el.tail for el in tree.xpath("//table[@class='data']//tr/td[9]/br")]
light = [el.tail for el in tree.xpath("//table[@class='data']//tr/td[10]/br")]
location = [el.tail for el in tree.xpath("//table[@class='data']//tr/td[11]/br")]
time = [el.tail for el in tree.xpath("//table[@class='data']//tr/td[12]/br")]
exif = [el.tail for el in tree.xpath("//table[@class='data']//tr/td[13]/br")]

info_dataframe = pd.DataFrame({
    "number": numbers,
    "subject": subjects,
    "light": light,
    "location": location,
    "time": time,
    "exif": exif,
    "original_path": '',
    "expert0_path": '',
    "expert1_path": '',
    "expert2_path": '',
    "expert3_path": '',
    "expert4_path": '',
})

os.makedirs(or_dir, exist_ok=True)
for expert_dir in expert_dirs:
    os.makedirs(expert_dir, exist_ok=True)

idxs = range(len(original_links))
if reverse:
	print("Downloading in reverse order")
	idxs = reversed(idxs)
	original_links = reversed(original_links)
	expert_links = reversed(expert_links)

for im_count, (original_link, expert_link) in zip(idxs, zip(original_links, expert_links)):
    print("Processing original image {}...".format(im_count), end="\r")
    filename = os.path.join(or_dir, '{}.png'.format(im_count))
    
    if not os.path.exists(filename):
        #Workaround for dng: first save dng, then convert to array, then to png
        image_link = os.path.join(base_link, original_link)
        response = requests.get(image_link, stream=True)
        with open('temp.dng', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        with rawpy.imread('temp.dng') as raw:
            rgb = raw.postprocess()
        del response
        #Clean temp dng
        os.remove('temp.dng')
        image = Image.fromarray(rgb)
        image.thumbnail((size, size), Image.ANTIALIAS)
        image.save(filename) 
        
    info_dataframe.at[im_count, 'original_path'] = filename 
        
    for expert_count, link in enumerate(expert_link):
        print("Processing image {} of expert {}...".format(im_count, expert_count), end="\r")
        filename = os.path.join(expert_dirs[expert_count], '{}.png'.format(im_count))
        
        if not os.path.exists(filename):
            #Download the image and resize to desired max size
            image_link = os.path.join(base_link, link)
            response = requests.get(image_link, stream=True)
            image = Image.open(response.raw)
            image.thumbnail((size, size), Image.ANTIALIAS)
            image.save(filename)
        info_dataframe.at[im_count, 'expert{}_path'.format(expert_count)] = filename   
    
    if limit_number != 0 and (im_count+1) == limit_number:
        break

info_dataframe.to_csv("mitdatainfo.csv", index=False)
