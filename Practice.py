import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,10,20)
y=x**1/2
plt.plot(x,y)
plt.show()

#to name the axis and title


plt.title('new plot')
plt.xlabel('X label')
plt.ylabel('Y label')


#creating multi-plots on the same fig using .subplot()

plt.subplot(1,2,1)
plt.plot(x,y,'red')
plt.subplot(1,2,2)
plt.plot(x,y, 'green')



#object oriented Interface-creating a figure onjects and calling methods off it

#creating a blank figure using .figure() method
fig=plt.figure()

#adding a set of axes using .add_axes()

ax=plt.add_axes([0.1,0.2,0.6,0.8])

#plotting the x and y arrays on it

ax.plot(x,y,'green')

#adding labels

ax.set_title('Plot uing Object Oriented Interface')
ax.set_xlabel('X Label')
ax.set_ylabel('Y label')

#multiple figure

fig=plt.figure()

ax1=fig.add_axes([0.1,0.2,0.7,0.9])
ax2=fig.add_axes([0.2,0.4,0.5,0.3])

ax1.plot(x,y,'purple')
ax2.plot(x,y,'green')

ax1.set_title('Plot uing Object Oriented Interface')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y label')

ax2.set_title('Plot uing Object Oriented Interface')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y label')


#creating multiplot  in Object Oriented interface uing .subplots() method

#creating a 4 by 4 subplots

fig, ax=plt.subplots(nrows=4,ncols=4)
plt.tight_layout() #to solve issue of overlapping

#plotting the x and y array on them

fig,ax=plt.subplots(nrows=4,ncols=4)
ax[0,0].plot(x,y 'green')
ax[2,3].plot(y,x,'red')
ax[0,2].plot(x,y,'purple')
plt.tight_layout()

#setting the title and the x and y labels

ax[0,0].set_title('First plot')
ax[0,0].set_xlabel('X  Label')
ax[0,0].set_ylabel('Y Label')

#figure size, aspect ratio, and DPI

fig=plt.figure(fig_size=(8,2),dpi=100)
ax=fig.add_axes([0,0,1,1])
ax.plot(x,y)

#for subplots

fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(7,4),dpi=100)
ax[0,0].plot(x,y)
ax[0,1].plot(y,x)
plt.tight_layout()

#saving a figure

fig.savefig('my_figure.png')

#displaying the saved figure

plt.imshow(mpimg.imread('my_figure.png'))

#decorating figures usimg legends

fig=plt.figure(figsize=(7,5),dpi=70)
ax=fig.add_axes([0,0,1,1])
ax.plot(x,x**2,'red',label='X square plot')
ax.plot(x,x**3,'green',label='X cube plot')
ax.legend()

#plot appearance-using linewidth(lw),linestyle(ls),marking out the data point using marker

fig=plt.figure(figsize=(7,5),dpi=50)
ax=fig.add_axes([0,0,1,1])
#changing the thickness to 3,linestyle to dashes,and marking out the data point
ax.plot(x,y,color='green',linewidth=3,linestyle='--',marker='x',markersize=7)

#plot range
fig=plt.figure(figsize=(8,6),dpi=60)
ax=fig.add_axes([0,0,1,1])
ax.plot(x,y,color='red',lw=3,ls='--')
ax.set_xlim([0,1])
ax.set_ylim([0,5])

#special plot types

#histogram-.hist() method

x=np.random.randn(1000)
plt.hist(x)

#time series(lineplot) 

import matplotlib.pyplot as plt
import datetime
import numpy as np

x=np.array([datetime.datetime(2018,9,i,0) for i in rangr(24)])
y=np.random.randint(100,size=x.shape)
plt.plot(x,y)
plt.show()

#scatter plots-.scatter() method

fig,ax=plt.subplots()
x=np.linspace(-1,1,50)
y=np.random.randn(50)
ax.scatter(x,y)

#bar graphs-.bar() method

import matplotlib.pyplot as plt
import pandas as  pd
import numpy as np

df=pd.DataFrame(np.random.rand(7,3),columns=['a','b','c','d'])
df.plot.bar()