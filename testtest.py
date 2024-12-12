import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
import threading as thr


flagStop = False


def testControlLoop(a):
    global flagStop
    while not flagStop:
        # pass
        print(a)

    print("The loop is stopped!")

    return


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("PyQt5 Dark Mode Example")
        self.setGeometry(100, 100, 400, 300)

        # Create a label, a text input field (QLineEdit), and a button
        self.label = QLabel("Enter your name:", self)
        self.label.setStyleSheet("color: white; font-size: 16px;")

        self.name_input = QLineEdit(self)
        self.name_input.setStyleSheet(
            "background-color: #333; color: white; padding: 5px; border-radius: 5px;"
        )

        self.greet_button = QPushButton("Greet Me!", self)
        self.greet_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;"
        )

        # Connect button click event to the greet function
        self.greet_button.clicked.connect(self.greet_user)

        self.red_button = QPushButton("Test Me!", self)
        self.red_button.setStyleSheet(
            "background-color: #AF4C50; color: white; padding: 10px; border-radius: 5px;"
        )

        # Create a layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.greet_button)
        layout.addWidget(self.red_button)

        # Set the layout for the window
        self.setLayout(layout)

        # Set the background color of the window to dark mode
        self.setStyleSheet("background-color: #121212;")

    def greet_user(self):
        global flagStop
        flagStop = True
        user_name = self.name_input.text()
        self.label.setText(f"Hello, {user_name}!")
        self.label.setStyleSheet("color: #FF9800; font-size: 16px;")


def main():
    # Initialize the application
    app = QApplication(sys.argv)

    # Create the main window
    window = MyWindow()

    window.show()

    thread = thr.Thread(target=testControlLoop, args=(5.0,))
    thread.start()

    # Run the application event loop
    sys.exit(app.exec_())

    thread.join()


if __name__ == "__main__":
    main()
