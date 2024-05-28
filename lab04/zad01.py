import math

def funcActivation(x):
    return 1 / (1 + math.exp(-x))

def forwardPass(wiek, waga, wzrost):
    hidden1 = wiek * -0.461220 + waga * 0.97314 + wzrost * -0.39203 + 0.80109
    hidden1_po_aktywacji = funcActivation(hidden1)
    hidden2 = wiek * 0.78548 + waga * 2.10684 + wzrost * -0.57847 + 0.43529
    hidden2_po_aktywacji = funcActivation(hidden2)
    output = hidden1_po_aktywacji * -0.81456 + hidden2_po_aktywacji * 1.03755 - 0.2368
    return output


print(forwardPass(23, 75, 176))
print(forwardPass(25, 67, 180))
