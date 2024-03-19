import serial
import wave
import matplotlib.pyplot as plt

# read in audio data
ser = serial.Serial('COM3', 115200)
array = []
i = 0
while(i < 16000):
    array.append(ser.readline())
    i += 1

# write to files
wav = wave.open("data.wav", "wb")
wav.setnchannels(1)
wav.setsampwidth(2)
wav.setframerate(4000)
y = []
for line in array:
    try:
        wav.writeframes(int(line.decode()[0:-2]).to_bytes(2, signed=True, byteorder='little'))
        y.append(int(line.decode()[0:-2]))
    except:
        print(f'Error: {line}')

# Plot the data
plt.plot(y)
plt.title('Data Plot')
plt.show()