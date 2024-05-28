import math
from datetime import datetime, timedelta

def calculate_wave(day):
    intellectual_wave = math.sin(math.pi * 2 * day / 33)
    emotional_wave = math.sin(math.pi * 2 * day / 28)
    physical_wave = math.sin(math.pi * 2 * day / 23)
    return intellectual_wave, emotional_wave, physical_wave

def calculate_days_lived(birth_date):
    today = datetime.now()
    difference = today - birth_date
    return difference.days

def next_day_comfort(birth_date):
    next_day = birth_date + timedelta(days=1)
    days_lived = calculate_days_lived(next_day)
    intelect, emotions, physical = calculate_wave(days_lived)
    next_average = (intelect + emotions + physical) / 3
    if next_average > 0.5:
        return "Nie martw się jutro będzie lepiej! :)"
    return None

def main():
    name = input("Podaj swoje imię: ")
    birth_day = int(input("Podaj dzień urodzenia: "))
    birth_month = int(input("Podaj miesiąc urodzenia: "))
    birth_year = int(input("Podaj rok urodzenia: "))

    birth_date = datetime(birth_year, birth_month, birth_day)
    current_day = calculate_days_lived(birth_date)

    intellectual_wave, emotional_wave, physical_wave = calculate_wave(current_day)

    print(f"Witaj {name}!")

    print(f"Twoje wartości intelektualnej fali: {intellectual_wave}")
    print(f"Twoje wartości emocjonalnej fali: {emotional_wave}")
    print(f"Twoje wartości fizycznej fali: {physical_wave}")

    if intellectual_wave > 0.5 and emotional_wave > 0.5 and physical_wave > 0.5:
        print("Gratulacje! Twoje wartości są wyższe niż 0.5.")
    elif intellectual_wave < -0.5 and emotional_wave < -0.5 and physical_wave < -0.5:
        print("Przyka sprawa.")
        next_day_msg = next_day_comfort(birth_date)
        if next_day_msg:
            print(next_day_msg)

if __name__ == "__main__":
    main()
