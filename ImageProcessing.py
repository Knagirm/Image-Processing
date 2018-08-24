from Tkinter import *
from PIL import Image,ImageTk
import tkFileDialog as filedialog
import os	
import numpy as np	
import matplotlib.pyplot as plt
import matplotlib
import math
gamma=0
stack = [] #to store img_t objects 
root = Tk(screenName=None,  baseName=None, className=' EE610_Assignment_1',  useTk=1)#creating window
root.resizable(width=True, height=True) 
img= Image.open('/home/mrigankash/Desktop/white.jpg')#initially Displayin a white image
img_t= ImageTk.PhotoImage(img)
panel= Label(root, image=img_t)
panel.pack()
def open_image():
	global img,img_t
	#path='/home/mrigankash/Documents/402712113-dark-forest-wallpapers.jpg'
	path=filedialog.askopenfilename(title='open')#asks user to open an image
	img = Image.open(path)  
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t)
	panel.configure(image=img_t)
	panel.image = img_t
	
def display():
	panel.configure(image=img_t)
	panel.image = img_t
def equalize():
	global img,img_t
	arr = np.array(img.convert("HSV"))
	m,n,l= arr.shape
	#equalize Histogram
	pix=np.zeros(shape=(256,1))
	pix_cdf=np.zeros(shape=(256,1))
	for x in range(arr.shape[0]):
		for y in range(arr.shape[1]):
			pix[arr[x,y,2],0]=pix[arr[x,y,2],0]+(1.0/(arr.shape[0]*arr.shape[1])) #calculating probability of each intensity
	pix_cdf[0,0]=pix[0,0]
	for x in range(1,255):
		pix_cdf[x,0]=pix_cdf[x-1,0]+pix[x,0]# cumlative density function of image intensity
	for x in range(0,255):
		pix_cdf[x,0]=round(pix_cdf[x,0]*255)
	new_arr=np.zeros(shape=arr.shape)
	for x in range(arr.shape[0]):
		for y in range(arr.shape[1]):
			new_arr[x,y,2]=pix_cdf[arr[x,y,2],0] #transforming back to image by mapping intensities
			new_arr[x,y,1]=arr[x,y,1]
			new_arr[x,y,0]=arr[x,y,0]
	arr_rgb=(matplotlib.colors.hsv_to_rgb(new_arr/255.0))*255.0 #converts HSV to RGB
	img= Image.fromarray(arr_rgb.astype('uint8'), 'RGB') #converts from numpy to image object
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t) #appends current object onto stack
	display()#image loading to root pending
def log_transform():
	global img,img_t
	arr = np.array(img.convert("HSV"))
	m,n,l= arr.shape
	#log Transform
	new_arr = np.zeros(shape=arr.shape)
	for x in range(arr.shape[0]):
		for y in range(arr.shape[1]):
			new_arr[x,y,2]=round(math.log(1+arr[x,y,2])*255.0/math.log(256))
			#new_arr[x,y,2]=255-arr[x,y,2]
			new_arr[x,y,1]=arr[x,y,1]
			new_arr[x,y,0]=arr[x,y,0]			
	arr_rgb=(matplotlib.colors.hsv_to_rgb(new_arr/255.0))*255.0
	img= Image.fromarray(arr_rgb.astype('uint8'), 'RGB')
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t)
	display()
def gamma_correct():
	global img,img_t,gamma
	gamma=2
	#input()
	print gamma
	arr = np.array(img.convert("HSV"))
	m,n,l= arr.shape
	new_arr = np.zeros(shape=arr.shape)
	for x in range(arr.shape[0]):
		for y in range(arr.shape[1]):
			new_arr[x,y,2]=round(math.pow(arr[x,y,2]/255.0,gamma))*255.0
			new_arr[x,y,1]=arr[x,y,1]
			new_arr[x,y,0]=arr[x,y,0]			
	arr_rgb=(matplotlib.colors.hsv_to_rgb(new_arr/255.0))*255.0
	img= Image.fromarray(arr_rgb.astype('uint8'), 'RGB')
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t)
	display()
def blurring():
	global img,img_t
	arr = np.array(img.convert("HSV"))
	m,n,l= arr.shape
	N = 5;
	k =N/2;
	new_arr=np.zeros(shape=(m, n, 3))
	weight= np.zeros(shape=(N,N))
	for x in range(N):
		for y in range(N):
			weight[x,y]=1.0 /(N*N);
	#we must pad with odd n =2k+1 i.e k 
	arr_pad=np.zeros(shape=(m+N-1,n+N-1))
	for x in range(k,m+k):
		for y in range(k,n+k):
			arr_pad[x,y]=arr[x-k,y-k,2]#padded matrix
	for x in range(m):
		for y in range(n):
			new_arr[x,y,2]=round(np.sum(np.multiply(weight,arr_pad[x:x+N,y:y+N])))
			new_arr[x,y,1]=arr[x,y,1]
			new_arr[x,y,0]=arr[x,y,0]
	arr_rgb=(matplotlib.colors.hsv_to_rgb(new_arr/255.0))*255.0
	img = Image.fromarray(arr_rgb.astype('uint8'), 'RGB')
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t)
	display()
def sharpening():
	global img,img_t
	arr = np.array(img.convert("HSV"))
	m,n,l= arr.shape
	N = 3;
	k=1
	new_arr=np.zeros(shape=(m, n, 3))
	weight= np.zeros(shape=(N,N))
	weight[1,0]=weight[0,1]=weight[2,1]=weight[1,2]=1
	weight[1,1]=-4
	'''weight[1,0]=weight[0,1]=weight[2,1]=weight[1,2]=weight[0,0]=weight[0,2]=weight[2,0]=weight[2,2]=1
	weight[1,1]=-8'''
	#we must pad with odd n =2k+1 i.e k 
	arr_pad=np.zeros(shape=(m+N-1,n+N-1))
	for x in range(k,m+k):
		for y in range(k,n+k):
			arr_pad[x,y]=arr[x-k,y-k,2]#padded matrix
	for x in range(m):
		for y in range(n):
			new_arr[x,y,2]=arr[x,y,2]-round(np.sum(np.multiply(weight,arr_pad[x:x+N,y:y+N])))
			new_arr[x,y,1]=arr[x,y,1]
			new_arr[x,y,0]=arr[x,y,0]
	arr_rgb=(matplotlib.colors.hsv_to_rgb(new_arr/255.0))*255.0
	img = Image.fromarray(arr_rgb.astype('uint8'), 'RGB')
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t)
	display()
def edge_detection():
	global img,img_t
	arr = np.array(img.convert("HSV"))
	m,n,l= arr.shape
	N = 3;
	k=1
	new_arr=np.zeros(shape=(m, n, 3))
	weight_1= np.zeros(shape=(N,N))
	weight_2= np.zeros(shape=(N,N))
	weight_1[0,0]=weight_1[0,2]=weight_2[0,0]=weight_2[2,0]=-1
	weight_1[2,0]=weight_1[2,2]=weight_2[0,2]=weight_2[2,2]=1
	weight_1[0,1]=weight_2[1,0]=-2
	weight_1[2,1]=weight_2[1,2]=2
	arr_pad=np.zeros(shape=(m+N-1,n+N-1))
	for x in range(k,m+k):
		for y in range(k,n+k):
			arr_pad[x,y]=arr[x-k,y-k,2]#padded matrix
	for x in range(m):
		for y in range(n):
			a=np.sum(np.multiply(weight_1,arr_pad[x:x+N,y:y+N]))
			b=np.sum(np.multiply(weight_2,arr_pad[x:x+N,y:y+N]))
			new_arr[x,y,2]=round((a*a + b*b)**(0.5))
			new_arr[x,y,1]=arr[x,y,1]
			new_arr[x,y,0]=arr[x,y,0]
	arr_rgb=(matplotlib.colors.hsv_to_rgb(new_arr/255.0))*255.0
	img = Image.fromarray(arr_rgb.astype('uint8'), 'RGB')
	img_t = img.resize((1280, 768)) #resizes image in screen
	img_t = ImageTk.PhotoImage(img_t)
	stack.append(img_t)
	display()
def undo():
	global img_t
	img_t=stack.pop()
	display()
def save():
	path=filedialog.asksaveasfilename(title='Save')
	img.save("{0}".format(path))

menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Load',command=open_image)#perfect
filemenu.add_command(label='Display',command=display)
filemenu.add_command(label='Equalise',command=equalize)
filemenu.add_command(label='Log Transform',command=log_transform)
filemenu.add_command(label='Gamma Transform',command=gamma_correct)
filemenu.add_command(label='Blurring',command=blurring)
filemenu.add_command(label='Sharpening',command=sharpening)
filemenu.add_command(label='Edge Detection',command=edge_detection)
filemenu.add_command(label='Undo',command=undo)
filemenu.add_command(label='Save',command=save)#,command=save_image)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=root.quit)
helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='About')
#Load an image using OpenCV
root.mainloop()


