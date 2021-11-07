import glob
import xml.etree.ElementTree as ET
import cv2
import os

def xml_read(file):
    tree=ET.parse(file)
    root=tree.getroot()
    object_file=root[6]
    bbox=object_file[4]
    xmin=bbox[0].text
    ymin=bbox[1].text
    xmax=bbox[2].text
    ymax=bbox[3].text
    print(xmin)
    print(ymin)
    print(xmax)
    print(ymax)
    return int(xmin),int(ymin),int(xmax),int(ymax)

def crop(root_folder,save_folder):
    files=sorted(glob.glob(root_folder+'/*.xml'))
    for file in files:
        base=file.split('.')[-2]
        base_name=base.split('/')[-1]
        new_name=save_folder+'/'+base_name+'.jpg'
        image=base+'.jpg'
        image_in=cv2.imread(image)

        xmin,ymin,xmax,ymax=xml_read(file)
        crop = image_in[ymin:ymax, xmin:xmax]
        cv2.imwrite(new_name, crop)
def sort_files(root_folder, save_folder):
    files=sorted(glob.glob(root_folder+'/*.jpg'))
    for file in files:
        base=file.split('(')[-2].rstrip()
        base_path=save_folder+'/'+base.split('/')[-1]
        base_name=file.split('/')[-1]
        try:
            os.system(f'mkdir "{base_path}"')
        except:
            print('existiert')
        save_path=base_path+'/'+base_name
        os.system(f"cp '{file}' '{save_path}'")
        
def rename(folder):
    files=glob.glob(folder+'/*.jpg')
    for file in files:
        try:
            liste_gesp=file.split(' ')
            new_name=''.join(liste_gesp)
            os.system(f"mv '{file}' '{new_name}'")
        except:
            print('existiert')
            
            
def move(root_folder, save_folder):
    folders=glob.glob(root_folder+'/*')
    for folder in folders:
        files=glob.glob(folder+'/*.jpg')
        files=files[0:5]
        folder_end=folder.split('/')[-1]
        for file in files:
            try:
                file_end=file.split('/')[-1]
                print('file_end',file_end)
                new_name=save_folder+'/'+folder_end+'/'+file_end
                os.system(f"mv '{file}' '{new_name}'")
            except:
                print('existiert')
#move('/home/florian/Desktop/hackathon2021/dataset_sorted/val','/home/florian/Desktop/hackathon2021/dataset_sorted/train')
#crop('/home/florian/Desktop/hackathon2021/dataset_raw/Test/', '/home/florian/Desktop/hackathon2021/dataset_cropped/Test/')
#xml_read('/home/florian/Desktop/hackathon2021/dataset_raw/Test/Woman(51).xml')
#sort_files('/home/florian/Desktop/hackathon2021/dataset_cropped/Test','/home/florian/Desktop/hackathon2021/dataset_sorted/val')