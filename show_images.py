import os
import shutil
import os.path as path
from collections import OrderedDict as odict
from PIL import Image, ImageTk
import random

import numpy as np

import tkinter as t
import tkinter.filedialog as fd

att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

paths = odict([
    ('Ours', './output/128/sample_testing')
])

rec_paths = odict([
    ('Ours', './output/128/sample_testing')
])

cat_dict = {
    'Ours': '4_Ours'
}


raw_img = 'Ours'

atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
        'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open',
        'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

atts_show = {
    'Bald': 'Bald',
    'Bangs': 'Bangs',
    'Black_Hair': 'Black_Hair',
    'Blond_Hair': 'Blond_Hair',
    'Brown_Hair': 'Brown_Hair',
    'Bushy_Eyebrows': 'Bushy_Eyebrows',
    'Eyeglasses': 'Eyeglasses',
    'Male': 'Male',
    'Mouth_Slightly_Open': 'Mouth_Open',
    'Mustache': 'Mustache',
    'No_Beard': 'No_Beard',
    'Pale_Skin': 'Pale_Skin',
    'Young': 'Young'
}
num_att = len(atts)

crop_dict = {
    'Ours': [(i*128+140, 0, i*128+268, 128) for i in range(1, num_att+1)]
}

crop_rec_dict = {
    'Ours': (140, 0, 268, 128)
}

# crop_stargan = [(i*128, 0, i*128+128, 128) for i in range(1, num_att+1)]
# crop_others = [(i*128+140, 0, i*128+268, 128) for i in range(1, num_att+1)]

# crop_stargan_rec = (128, 0, 256, 128)
# crop_others_rec = (140, 0, 268, 128)

lmt_low = 182638
lmt_high = 202599
img_nums = list(range(lmt_low, lmt_high+1))


att_file = '/home/data/Datasets/CelebA/Img/list_attr_celeba.txt'
att_cols = [att_dict[i]+1 for i in atts]
attr = np.loadtxt(att_file, skiprows=2, usecols=att_cols, dtype=np.str)

att_dict = {atts[i]:i for i in range(len(atts))}
att_dict['rec'] = -1

class MyWindow:
    
    def __init__(self):
        self.root = t.Tk(className='Face Attribute Manipulation')
        self.root.bind('<Right>', self.nxt)
        self.root.bind('<Left>', self.prev)
        self.root.bind('<Down>', self.nxt)
        self.root.bind('<Up>', self.prev)
        self.root.bind('<F5>', self.refresh)
        
        gray = Image.new('RGB', (128, 128), color='gray')
        self.grayTkimg = ImageTk.PhotoImage(gray)
        
        
        
        self.can = t.Frame(self.root)
        
        self.cur_img = str(img_nums[0])
        self.cur_imgs = self.process_img(self.cur_img)
        
        ######### row 0 #########
        self.raw_img = ImageTk.PhotoImage(self.cur_imgs['raw'])
        self.row0 = t.Canvas(self.can)
        self.label_raw_img = t.Label(self.row0, image=self.raw_img)
        self.label_raw_img.pack(side=t.LEFT)
        
        bt1 = t.Canvas(self.row0)
        t.Button(bt1, text=u'random', command=self.random, width=10).pack(side=t.RIGHT)
        self.num = t.StringVar(value=self.cur_img)
        t.Entry(bt1, textvariable=self.num, bg='white').pack(side=t.RIGHT)
        bt1.pack(anchor=t.E)
        
        bt2 = t.Canvas(self.row0)
        t.Button(bt2, text=u'refresh', command=self.refresh, width=10).pack(side=t.RIGHT)
        t.Button(bt2, text=u'next', command=self.nxt, width=10).pack(side=t.RIGHT)
        t.Button(bt2, text=u'prev', command=self.prev, width=10).pack(side=t.RIGHT)
        bt2.pack(side=t.BOTTOM, anchor=t.E)
        
        atts_can1 = t.Canvas(self.row0)
        for i in range(num_att//2):
            setattr(self, atts[i], t.BooleanVar(value=True))
            t.Checkbutton(atts_can1, text=atts[i], variable=getattr(self, atts[i]), width=20, anchor=t.W).pack(side=t.LEFT)
        atts_can1.pack(anchor=t.W, side=t.TOP)
        atts_can2 = t.Canvas(self.row0)
        for i in range(num_att//2, num_att):
            setattr(self, atts[i], t.BooleanVar(value=True))
            t.Checkbutton(atts_can2, text=atts[i], variable=getattr(self, atts[i]), width=20, anchor=t.W).pack(side=t.LEFT)
        atts_can2.pack(anchor=t.W, side=t.TOP)
        self.row0.pack(fill=t.X, side=t.TOP, anchor=t.CENTER)
        
        for cat in paths:
            setattr(self, 'can_%s'%(cat), t.Canvas(self.can))
            #
            setattr(self, 'img_%s_rec'%cat, ImageTk.PhotoImage(self.cur_imgs[cat][-1]))
            setattr(self, 'lbl_%s_rec'%cat, t.Label(getattr(self, 'can_%s'%cat), image=getattr(self, 'img_%s_rec'%cat)))
            getattr(self, 'lbl_%s_rec'%cat).pack(side=t.LEFT, padx=2)
            for i in range(num_att):
                setattr(self, 'img_%s_%s'%(cat, atts[i]), ImageTk.PhotoImage(self.cur_imgs[cat][i]))
                setattr(self, 'lbl_%s_%s'%(cat, atts[i]), t.Label(getattr(self, 'can_%s'%(cat)), image=getattr(self, 'img_%s_%s'%(cat, atts[i]))))
                getattr(self, 'lbl_%s_%s'%(cat, atts[i])).pack(side=t.LEFT, padx=2)
            getattr(self, 'can_%s'%(cat)).pack(fill=t.X)
        
        row_att = t.Canvas(self.can)
        self.Label_rec = t.Button(row_att, text='label_rec', command=self.save_rec, width=18)
        self.Label_rec.pack(side=t.LEFT)
        for att in atts:
            if attr[int(self.num.get())-1][att_dict[att]] == '1':
                setattr(self, 'label_%s'%att, '---'+atts_show[att]+'---')
            else:
                setattr(self, 'label_%s'%att, '+++'+atts_show[att]+'+++')
            setattr(self, 'Label_%s'%att, t.Button(row_att, text=getattr(self, 'label_%s'%att), 
                                                   command=getattr(self, 'save_%s'%att), width=18))
            getattr(self, 'Label_%s'%att).pack(side=t.LEFT)
        row_att.pack(fill=t.X)
        self.can.pack(fill=t.BOTH)
        
        window_width = (128+8)*(num_att+1)
        window_height = 128*(len(paths)+1)+30
        window_pos_x = (self.root.winfo_screenwidth()-window_width)/2
        window_pos_y = (self.root.winfo_screenheight()-window_height)/2
        self.root.geometry('%dx%d+%d+%d'%(window_width, window_height, window_pos_x, window_pos_y))
        self.root.mainloop()

    def process_img(self, num):
        imgs = {}
        for cat in paths:
            try:
                wimg = Image.open(path.join(paths[cat], str(num)+'.png'))
            except:
                try:
                    wimg = Image.open(path.join(paths[cat], str(num)+'.jpg'))
                except:
                    print('Can\'t open %s of %s'%(str(num), cat))
                    continue
            if cat == raw_img:
                rimg = wimg.crop((0, 0, 128, 128))
                imgs['raw'] = rimg
            imgs[cat] = [wimg.crop(crop_dict[cat][i]) for i in range(num_att)]
            # if cat == 'StarGAN':
            #     imgs[cat] = [wimg.crop(crop_stargan[i]) for i in range(num_att)]
            # else:
            #     imgs[cat] = [wimg.crop(crop_others[i]) for i in range(num_att)]
            wimg.close()
        for cat in rec_paths:
            try:
                wimg = Image.open(path.join(rec_paths[cat], str(num)+'.png'))
            except:
                try:
                    wimg = Image.open(path.join(rec_paths[cat], str(num)+'.jpg'))
                except:
                    print('Can\'t open %s of rec_%s'%(str(num), cat))
            imgs[cat].append(wimg.crop(crop_rec_dict[cat]))
            # if cat == 'StarGAN':
            #     imgs[cat].append(wimg.crop(crop_stargan_rec))
            # else:
            #     imgs[cat].append(wimg.crop(crop_others_rec))
            wimg.close()
        return imgs
                
    def random(self):
        self.show_img(str(random.choice(img_nums)))
            
    def show_img(self, num):
        self.num.set(num)
        imgs = self.process_img(num)
#        print(imgs)
        self.raw_img = ImageTk.PhotoImage(imgs['raw'])
        self.label_raw_img.config(image=self.raw_img)
        for cat in paths:
            setattr(self, 'img_%s_rec'%cat, ImageTk.PhotoImage(imgs[cat][-1]))
            getattr(self, 'lbl_%s_rec'%cat).config(image=getattr(self, 'img_%s_rec'%cat))
            for i in range(num_att):
                if not getattr(self, atts[i]).get():
                    setattr(self, 'img_%s_%s'%(cat, atts[i]), self.grayTkimg)
                    getattr(self, 'lbl_%s_%s'%(cat, atts[i])).config(image=self.grayTkimg)
                    continue
                setattr(self, 'img_%s_%s'%(cat, atts[i]), ImageTk.PhotoImage(imgs[cat][i]))
                getattr(self, 'lbl_%s_%s'%(cat, atts[i])).config(image=getattr(self, 'img_%s_%s'%(cat, atts[i])))
#                getattr(self, 'lbl_%s_%s'%(cat, atts[i])).image = ImageTk.PhotoImage(imgs[cat][i])
        for att in atts:
            if attr[int(self.num.get())-1][att_dict[att]] == '1':
                setattr(self, 'label_%s'%att, '---'+atts_show[att]+'---')
            else:
                setattr(self, 'label_%s'%att, '+++'+atts_show[att]+'+++')
            getattr(self, 'Label_%s'%att).config(text=getattr(self, 'label_%s'%att))
    def refresh(self, event=None):
        num = int(self.num.get())
        if num < lmt_low:
            self.num.set(lmt_low)
            num = lmt_low
        elif num > lmt_high:
            self.num.set(lmt_high)
            num = lmt_high
        self.show_img(str(num))
        
    def prev(self, event=None):
        self.num.set(str(int(self.num.get())-1))
        self.refresh()
        
    def nxt(self, event=None):
        self.num.set(str(int(self.num.get())+1))
        self.refresh()

    def save(self, att=None, event=None):
        save_root = 'selected_images'
        try:
            os.mkdir(save_root)
        except:
            pass
        save_dir = '%s_%s'%(self.num.get(), att)
        try:
            os.mkdir(path.join(save_root, save_dir))
        except:
            pass
        imgs = self.process_img(self.num.get())
        imgs['raw'].save(path.join(save_root, save_dir, '0_raw.png'), 'png')
        cnt = 0
        for cat in paths:
            img = imgs[cat][att_dict[att]]
            cnt += 1
            img.save(path.join(save_root, save_dir, '%d_%s.png'%(cnt, cat)), 'png')

    def save_rec(self):
        self.save('rec')
    def save_Bald(self):
        self.save('Bald')
    def save_Bangs(self):
        self.save('Bangs')
    def save_Black_Hair(self):
        self.save('Black_Hair')
    def save_Blond_Hair(self):
        self.save('Blond_Hair')
    def save_Brown_Hair(self):
        self.save('Brown_Hair')
    def save_Bushy_Eyebrows(self):
        self.save('Bushy_Eyebrows')
    def save_Eyeglasses(self):
        self.save('Eyeglasses')
    def save_Male(self):
        self.save('Male')
    def save_Mouth_Slightly_Open(self):
        self.save('Mouth_Slightly_Open')
    def save_Mustache(self):
        self.save('Mustache')
    def save_No_Beard(self):
        self.save('No_Beard')
    def save_Pale_Skin(self):
        self.save('Pale_Skin')
    def save_Young(self):
        self.save('Young')

if __name__ == '__main__':
    mywin = MyWindow()
                
                
                
                
                
                
                
                
                
