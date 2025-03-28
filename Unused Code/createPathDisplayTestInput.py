

inputList = []

numOfInputs = 1000
hole_x_Location = 1000
hole_y_Location = 700

ball_x = 0
ball_y = 0

for i in range(numOfInputs):

    ball_x += 1
    ball_y += 1

    if ball_x >= 1440:
        ball_x = 0

    if ball_y >= 720:
        ball_y = 0

    inputList.append([ball_x, ball_y, hole_x_Location, hole_y_Location])


with open("test_path_display_inputs.txt", "w") as f:
    for x in inputList:
        f.write(str(x[0]) + " ")
        f.write(str(x[1]) + " ")
        f.write(str(x[2]) + " ")
        f.write(str(x[3]))
        f.write("\n")
