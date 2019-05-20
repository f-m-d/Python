#This program ask you to insert your name and your age

def main():
    print("Hello world!")
    print("What is your name?")
    myName=input()
    print("Nice to meet you, " + myName)


    print("Your name length is: " + str(len(myName)))
    print("How old are you?")
    age= input()
    print("You will be " + str(int(age)+1) + " next year")


main()