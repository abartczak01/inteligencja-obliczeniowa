import math
import numpy as np
import matplotlib.pyplot as plt

G = 9.81

def rzut_ukosny(kat, h, v):
    kat_rad = math.radians(kat)
    d = v * math.cos(kat_rad) * (v * math.sin(kat_rad) + math.sqrt((v * math.sin(kat_rad)) ** 2 + 2 * G * h)) * 1 / G
    return d

def stworz_wykres_rzutu(kat, h, v):
    kat_rad = np.deg2rad(kat)
    total_time = (v * np.sin(kat_rad) + np.sqrt((v * np.sin(kat_rad)) ** 2 + 2 * G * h)) / G
    time = np.linspace(0, total_time, num=500)
    height = h + (v * np.sin(kat_rad) * time) - (0.5 * G * (time ** 2))
    range_x = v * np.cos(kat_rad) * time

    plt.plot(range_x, height)
    plt.title('Projectile Motion for the Trebuchet')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.savefig('trajektoria.png')

def main():
    h = 100
    v = 50
    
    cel = np.random.randint(50, 340)
    print(f"Cel znajduje się {cel} metrów od miejsca strzału.")
    rundy = 0
    
    while True:
        try:
            rundy += 1
            kat = float(input("Podaj kąt strzału (w stopniach): "))
            odleglosc = rzut_ukosny(kat, h, v)
            blad = abs(odleglosc - cel)
            print(f"Strzeliłeś na odległość: {odleglosc:.2f} metrów")
            if blad <= 5:
                print(f"Cel trafiony! Liczba prób: {rundy}")
                stworz_wykres_rzutu(kat, h, v)
                break
            else:
                print(f"Chybienie o: {blad:.2f} metrów. Spróbuj ponownie.")
        except ValueError:
            print("Błąd: Podana wartość nie jest liczbą.")


main()
