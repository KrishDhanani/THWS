# Numpy related things
import numpy as np

# 1. Shapes:
x = np.zeros((2,3)) # row * column
print(x)
print(f"Shape of X {x.shape}")

y = np.random.rand(2,3) # row * column 
# Also it only return random number between 0 & 1 due to use of "rand"
print(y)
print(f"Shape of Y {y.shape}")

z = np.random.randn(3,2,3) # num. of req. metrix * row * column 
# Due to use of "randn" it also return the negative number 
print(z)
print(f"Shape of Z {z.shape}")




# 2. Reshaping:
a = np.array([[-1, 3, 2], 
              [4, 5, 6]])
b = a.flatten() # making it flate
print(type(b))
print(b)

c = b.reshape(2,3)
print(f" C shape {c.shape}")