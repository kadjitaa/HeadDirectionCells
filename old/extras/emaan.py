# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 23:29:26 2020

@author: kasum
"""


def isNumeric(x):
    if x.isnumeric():
        return True
    else:
        xlist=x.split(".")
        if len(xlist) == 2 and xlist[0].isnumeric() and xlist[1].isnumeric():
            return True
        else:
            return False
# complete the program by writing your own code here

def isCorrect(x):
    x=x.lower()
    if x == 'yes':
        return x
    elif x=='no':
        return x
    else:
        print('invalid')

try:
    a=isCorrect(input('livedoid rush:' ).lower())
    if a=='yes':
        bodytemp=str(input('body temp:'))
        isNumeric(bodytemp)
        if float(bodytemp)>=38:
            ADA2=int(input('ADA2:'))
            if ADA2==0:
                print('low risk')
            elif ADA2==1:
                print('med risk')
            elif ADA2==2:
                print('high risk')
        elif float(bodytemp)<38:
            cpr=float(input('cpr:'))
            if cpr<5:
                print('low risk')
            elif cpr>=5:
                ADA2=int(input('ADA2:'))
                if ADA2==0:
                    print('low risk')
                elif ADA2==1:
                    print('med risk')
                elif ADA2==2:
                    print('high risk')
    elif a=='no':
        neuro=input('neuro dis:').lower()
        if neuro=='no':
            print('low risk')
        elif neuro =='yes':
            bodytemp=float(input('body temp:'))
            if bodytemp>=38:
                ADA2=int(input('ADA2:'))        
                if ADA2==0:
                    print('low risk')
                elif ADA2==1:
                    print('med risk')
                elif ADA2==2:
                    print('high risk')
            elif bodytemp<38:
                cpr=float(input('cpr:'))
                if cpr<5:
                    print('low risk')
                elif cpr>=5:
                    ADA2=int(input('ADA2:'))
                    if ADA2==0:
                        print('low risk')
                    elif ADA2==1:
                        print('med risk')
                    elif ADA2==2:
                        print('high risk')
except:
    print('inavlid')
    






if (liveSkinRash == "Yes" or liveSkinRash == "Y" or liveSkinRash == "yes" or liveSkinRash == "y"):
    bodytemp = float(input("Body temp (in C)?"))
    if bodytemp>=38:
        ADA2 = int(input("ADA2 pathogenic mutations?"))
        if ADA2 == 0:
            print("low-risk")
        elif ADA2 == 1:
            print("medium-risk")
        elif ADA2 == 2:
            print("high-risk")
        else:
            print("Invalid")
    elif bodytemp<38:
        CRP = int(input("CRP (mg/dL)?"))
        if CRP<5:
            print("low-risk")
        elif CRP>= 5:
            ADA2 = int(input("ADA2 pathogenic mutations?"))
            if ADA2 == 0:
                print("low-risk")
            elif ADA2 == 1:
                print("medium-risk")
            elif ADA2 == 2:
                print("high-risk")
            else:
                print("Invalid")
        else:
            print("Invalid")
    else:
        print("Invalid")        
elif (liveSkinRash == "No" or liveSkinRash == "N" or liveSkinRash == "no" or liveSkinRash == "n"):
    nd = input("Neurological disorder (Yes or No)? ")
    if (nd == "No" or nd == "N" or nd == "no" or nd == "n"):
      print("low-risk")  
    elif (nd == "Yes" or nd == "Y" or nd == "yes" or nd == "y"):
        bodytemp = float(input("Body temp (in C)?"))
        if bodytemp>=38:
            ADA2 = int(input("ADA2 pathogenic mutations?"))
            if ADA2 == 0:
                print("low-risk")
            elif ADA2 == 1:
                print("medium-risk")
            elif ADA2 == 2:
                print("high-risk")
            else:
                print("Invalid")
        elif bodytemp<38:
            CRP = int(input("CRP (mg/dL)?"))
            if CRP<5:
                print("low-risk")
            elif CRP>= 5:
                ADA2 = int(input("ADA2 pathogenic mutations?"))
                if ADA2 == 0:
                    print("low-risk")
                elif ADA2 == 1:
                    print("medium-risk")
                elif ADA2 == 2:
                    print("high-risk")
                else:
                    print("Invalid")
            else:
                print("Invalid")
        else:
            print("Invalid")
    else:
        print("Invalid")
else:
    print("Invalid")
    
    
    

liveSkinRash = input("Livedoid skin rash (Yes or No)? ")

def isNumeric(x):
    if x.isnumeric():
        return True
    else:
        xlist=x.split(".")
        if len(xlist) == 2 and xlist[0].isnumeric() and xlist[1].isnumeric():
            return True
        else:
            return False
# complete the program by writing your own code here







import numpy as np
tape=np.zeros(100)
idx=50
state='A'
print('current state: ' +state)

for i in range(len(tape)):
    symbol=tape[idx]
    if symbol == 0 and state=='A':
        tape[idx]=1
        idx-=1
        state='B'
        print('current state: ' +state)    
    elif symbol==0 and state=='B':
        tape[idx]=1
        idx+=1
        state='A'
        print('current state: ' +state)
    elif symbol==1 and state=='A':
        idx+=1
        state='C'
        print('current state: ' +state)   
    elif symbol==0 and state=='C':
        tape[idx]=1
        idx+=1
        state='B'
        print('current state: '+state)    
    elif symbol==1 and state=='B':
        idx-=1
        state='B'
        print('current state: '+state) 
        
    elif symbol==1 and state=='C':
        idx+=1
        state='H'
        print('current state: '+state)
        
        

                
         
         
        
        
    if symbol==0 and state=='B':
        print(str(i)+'current state: ' + state)
        tape[idx]=1
        idx-=1
        state='A'          
    if symbol==1 and state=='A':
        print(str(i)+'current state: ' + state)
        idx-=1
        state='C'
    if symbol==0 and state=='C':
        print(str(i)+'current state: ' + state)
        tape[idx]=1
        idx-=1
        state='B'
            
            
        elif symbol==0 and state=='C':
            tape[idx]=1
            idx-=1
            state='B'
           
        
        
        
    elif symbol==0 and state=='B':
        tape[idx]=1
        idx-=1
        state='A'
    elif symbol==1 and state=='A':
        idx+=1
        state='C'
    elsif
        
    
























tape=[]
ini_s=['A']
t_s=['H']
symbols=[0,1]










tape=[0]
if tape==0:
    print('move right')
    A=1
    print ('Current state: A' )
    
   

#1_decision tree


try:
    a=input('livedoid rush:' ).lower()
    if a not in ['yes', 'no']:
        print('invalid')

    if a=='yes':
        #bodytemp=float(input('body temp:'))
        bodytemp=str(input('body temp:'))
        if isNumeric(bodytemp):
            if float(bodytemp)>=38:
                ADA2=float(input('ADA2:'))
                if ADA2==0:
                    print('low risk')
                elif ADA2==1:
                    print('med risk')
                elif ADA2==2:
                    print('high risk')
            elif float(bodytemp)<38:
                cpr=float(input('cpr:'))
                if cpr<5:
                    print('low risk')
                elif cpr>=5:
                    ADA2=float(input('ADA2:'))
                    if ADA2==0:
                        print('low risk')
                    elif ADA2==1:
                        print('med risk')
                    elif ADA2==2:
                        print('high risk')
except:
    print('inavlid')  



                 
    elif a=='no':
        neuro=input('neuro dis:').lower()
        if neuro=='no':
            print('low risk')
        elif neuro =='yes':
            bodytemp=float(input('body temp:'))
            if bodytemp>=38:
                ADA2=float(input('ADA2:'))        
                if ADA2==0:
                    print('low risk')
                elif ADA2==1:
                    print('med risk')
                elif ADA2==2:
                    print('high risk')
            elif bodytemp<38:
                cpr=float(input('cpr:'))
                if cpr<5:
                    print('low risk')
                elif cpr>=5:
                    ADA2=float(input('ADA2:'))
                    if ADA2==0:
                        print('low risk')
                    elif ADA2==1:
                        print('med risk')
                    elif ADA2==2:
                        print('high risk')
except:
    print('invalid')
    
    
    
    
    
#####2
a=input('Enter a DNA sequence:')    
a='AABAAABA'
v=[]
for i in range(len(a)-2):
    v.append(a.index('AB',i))
unique(v)
    
    
count=0    
for i in range(len(a)):   
    x=a.index('ABA',6)
    count+=x
count=0    
while count>len(a):
    count+=1
    c=a.index("AB",count)
    print(c)
    "AB" in a
    
    
    
    
    
    