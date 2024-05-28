from datetime import datetime
import math

def calculate_days_lived(year, month, day):
    date_of_birth = datetime(int(year), int(month), int(day))
    today = datetime.now()
    difference = today - date_of_birth
    return difference.days

def wave(day, period):
    angle = math.pi * 2 * day / period
    return math.sin(angle)

def next_day_comfort(days_lived):
    intelect = wave(days_lived + 1, 33)
    emotions = wave(days_lived + 1, 28)
    physical = wave(days_lived + 1, 23)
    next_average = (intelect + emotions + physical) / 3
    if next_average > 0.5:
        return True
    return False

def get_wave_comment(intellect, emotions, physical, days_lived):
    average = (intellect + emotions + physical) / 3
    if average > 0.5:
        return "Congratulations, your waves are waving the awesome way!"
    elif average < -0.5:
        msg = "I'm sorry :(. Your waves are not that great today"
        if next_day_comfort(days_lived):
            msg += "\nDon't worry. Tomorrow's gonna be better."
        return msg
    return None

def main():
    name = input("Enter your name: ")
    year, month, day = "", "", ""
    while not all(map(str.isdigit, [year, month, day])) or not (0 < int(month) <= 12 and 0 < int(day) <= 31):
        year = input("Enter the year of your birth: ")
        month = input("Enter the month of your birth (in numeric format): ")
        day = input("Enter the day of your birth (in numeric format): ")

    print(f"Welcome, {name}!")
    dob = datetime(int(year), int(month), int(day))
    print("Name:", name)
    print("Date of birth:", dob.strftime("%d/%m/%Y"))

    days_lived = calculate_days_lived(year, month, day)
    print(f"You are currently experiencing your {days_lived} day of life.")

    intelect = wave(days_lived, 33)
    emotions = wave(days_lived, 28)
    physical = wave(days_lived, 23)

    print(f"Your waves:\n- Intellectual wave: {intelect}\n- Emotional: {emotions}\n- Physical: {physical}")

    wave_comment = get_wave_comment(intelect, emotions, physical, days_lived)
    if wave_comment:
        print(wave_comment)

main()
