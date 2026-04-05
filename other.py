import os
print("Current working directory is:", os.getcwd())
path = r"C:\Users\metin\PycharmProjects\Project_Kapo"
os.chdir(path)
print(r"Current working directory is:", os.getcwd())