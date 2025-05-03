import numpy as np

# Define constants and initialize variables
IN = 40
JN = 40
KN = 10
DENSIT = 995.7
VISCOS = 801e-6

# Initialize arrays
X = np.zeros(IN)
Y = np.zeros(JN)
Z = np.zeros(KN)
U = np.zeros((IN, JN, KN))
V = np.zeros((IN, JN, KN))
W = np.zeros((IN, JN, KN))

# Define functions for subroutines

def user_option():
    global X, Y, Z
    for i in range(1, IN):
        X[i] = X[i - 1] + 0.1
        Y[i] = Y[i - 1] + 0.1
    for i in range(1, KN):
        Z[i] = Z[i - 1] + 0.1

def mainpr():
    print("Running MAINPR subroutine...")
    # Placeholder for MAINPR logic

# Main program logic
def main():
    user_option()
    print("--------  output is in OO file  --------------")
    mainpr()

    # Write results to a file
    with open('raw.dat', 'w') as f:
        f.write(" ".join(map(str, X)) + "\n")
        f.write(" ".join(map(str, Y)) + "\n")
        f.write(" ".join(map(str, Z)) + "\n")
        # Placeholder for writing U, V, W arrays

if __name__ == "__main__":
    main()