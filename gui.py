import sys
from PyQt4 import QtGui, QtCore
import run as run

#class Widget(QtGui.QWidget):
class Widget(QtGui.QMainWindow):
    def __init__(self, app):
        self.V = app.desktop().screenGeometry()
        self.w = self.V.width() * 1/3
        self.h = self.V.height() * 3/4
 
        self.layout1 = QtGui.QGridLayout() #QtGui.QVBoxLayout()
        self.layout2 = QtGui.QGridLayout()
        self.layout3 = QtGui.QGridLayout()

        super(Widget, self).__init__()
        self.initTabs()
        self.initUI()

    def initTabs(self):
        pass

    def initUI(self): 
        #Window initialization
        #self.window = QtGui.QMainWindow()
        self.notebook = QtGui.QTabWidget()
        #self.notebook.setStyleSheet('{height: 100px; width: 100px;}')
        #self.notebook.setStyleSheet("QTabWidget().tabBar{ min-width: 100px; }");
        
        vBoxlayout1 = QtGui.QVBoxLayout()
        vBoxlayout2 = QtGui.QVBoxLayout()
        vBoxlayout3 = QtGui.QVBoxLayout()

        page1 = QtGui.QWidget()
        page2 = QtGui.QWidget()
        page3 = QtGui.QWidget()

        self.notebook.tabBar().setStyleSheet("QTabBar::tab{width:150px;}")
        self.notebook.tabBar().setExpanding(True)
        
        #self.notebook.setTabsClosable(True)
        self.notebook.tabBar().setMovable(True)
        #self.notebook.tabCloseRequested.connect(self.close_handler)
        self.setCentralWidget(self.notebook)
        
        self.notebook.setWindowTitle('Tabs')
        #self.notebook.setCurrentIndex(1)  

        #Button 1, a normal button
        QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))
        self.btn = QtGui.QPushButton('Generate!', self)
        self.btn.clicked.connect(self.starr)
        self.btn.move(self.w/2.50, self.h/1.35)  #btn.move(self.w/2.15, self.h/2.15) is middle 
        self.layout1.addWidget(self.btn)

        #Button 2, a quit button
        self.qbtn = QtGui.QPushButton('Quit', self)
        self.qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.qbtn.resize(self.qbtn.sizeHint())
        self.qbtn.move(self.w/1.35, self.h/1.1)
        self.layout1.addWidget(self.qbtn)

        #Text box, telling the user what to do
        self.instructions = QtGui.QLabel("This project involves a variational autoencoder, a popular\n approach to unsupervised learning.\r\n \
                                          General autoencoders rely on two networks, an\n encoder and a decoder which work together to \n \
                                          realize which parts of an image are the most important.\n This is accomplished through the \n \
                                          encoder, which reduces the dimensionality of an\n input image to reach its latent space rep-\n\
                                          resentation. It is the job of the decoder to try to\n reconstruct the original image from the \n \
                                          reduced data as best as it can. A variational\n autoencoder has an additional constraint; that \n \
                                          its latent vectors follow a Gaussian (normal)\n distribution, in order to aid in building a generative\n \
                                          model. \n \
                                          Our VAE is trained on the set of faces that Linus\n Sebastian (LinusTechTips) has made, and is able \n\
                                          to map faces of other humans onto his face as well.\n Upload an image and see! \n", self)
        self.instructions.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.instructions.setStyleSheet('QLabel { font-size: 10pt; color: #000000; }')
        #self.editor.textCursor().insertHtml('Upload an Image!')
        self.instructions.move(self.w/5, self.h/12)
        self.instructions.resize(380,70)
        #self.editor.setReadOnly(True)self.imageChanger = ImageChanger(images)
        #textbox.textCursor().insertHtml('<b>bold text</b>')
        self.layout2.addWidget(self.instructions)

        
        class ImageChanger(QtGui.QWidget):    
            def __init__(self, images, parent=None):
                super(ImageChanger, self).__init__(parent)        

                self.comboBox = QtGui.QComboBox(self)
                self.comboBox.addItems(images)

                self.layout = QtGui.QVBoxLayout(self)
                self.layout.addWidget(self.comboBox)

        

        #Instructions, for the first tab
        self.editor = QtGui.QLabel("Upload your favorite picture of yourself, and watch\nas your face is magically replaced by LinusTechTips'!", self)
        self.editor.setAlignment(QtCore.Qt.AlignCenter)
        self.editor.setStyleSheet('QLabel { font-size: 12pt; color: #000000; }')
        self.editor.move(self.w/16.25, self.h/12)
        self.editor.resize(400,50)
        self.layout1.addWidget(self.editor)  

        #self.imageChanger = ImageChanger(images)
        #self.imageChanger.move(self.imageChanger.pos().y(), self.imageChanger.pos().x() + 100)
        #self.imageChanger.show()
        #self.imageChanger.comboBox.currentIndexChanged[str].connect(self.changeImage)

        #Button 3, an image-pushing button 
        self.ibtn = QtGui.QPushButton(self)
        self.ibtn.setStyleSheet('QPushButton {self.notebook = QTabWidget() font-size: 45pt;}')
        self.ibtn.setText('Import Image')
        self.ibtn.clicked.connect(self.handleButton)
        self.ibtn.move(self.w/4.68, self.h/5)
        self.ibtn.resize(self.w/1.75, self.h/2)
        self.layout1.addWidget(self.ibtn)

        #Popup when the credits button is pressed
        class MyDialog(QtGui.QDialog):
            def __init__(self, parent=None):
                class A: 
                    pass

                super(MyDialog, self).__init__(parent)

                self.buttonBox = QtGui.QDialogButtonBox(self)
                self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
                self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Ok)
                self.buttonBox.clicked.connect(MyDialog.accept)
                self.buttonBox.accepted.connect(self.accept)
                self.buttonBox.rejected.connect(MyDialog.reject)

                self.textBrowser = QtGui.QTextBrowser(self)
                self.textBrowser.append("Credits:")
                self.textBrowser.append("Starr Yang")
                self.textBrowser.append("Nithin Raghavan")
                self.textBrowser.append("Rebecca Zeng")
                self.textBrowser.append("Tom Cheah")

                self.verticalLayout = QtGui.QVBoxLayout(self)
                self.verticalLayout.addWidget(self.textBrowser)
                self.verticalLayout.addWidget(self.buttonBox)
 
        #Button 4, a credits button
        self.cbtn = QtGui.QPushButton('Credits', self)
        self.cbtn.clicked.connect(self.on_pushButton_clicked)
        self.cbtn.resize(self.qbtn.sizeHint())
        self.cbtn.move(self.w/14, self.h/1.1)
        self.layout1.addWidget(self.cbtn)

        self.dialogTextBrowser = MyDialog(self)

        #Rendering the image from Generate
        pic.setGeometry(10, 10, 400, 200)
        pixmap = QtGui.QPixmap(FILENAME)
        pixmap = pixmap.scaledToHeight(200)
        pic.setPixmap(pixmap)

        #Creating page 1
        vBoxlayout1.addWidget(self.instructions)
        page1.setLayout(vBoxlayout1)

        #Creating page 2
        vBoxlayout2.addWidget(self.cbtn)
        vBoxlayout2.addWidget(self.editor)
        vBoxlayout2.addWidget(self.ibtn)
        vBoxlayout2.addWidget(self.btn)
        vBoxlayout2.addWidget(self.qbtn)
        page2.setLayout(vBoxlayout2)

        #Creating page 3

        self.notebook.addTab(page1, "Instructions")
        self.notebook.addTab(page2, "Upload")
        self.notebook.addTab(page3, "Results")

        self.notebook.setCurrentIndex(1)  
        self.notebook.show()
        
        #self.ibtn.setIcon(QtGui.QIcon('linustechtips.jpg'))
        #self.ibtn.setIconSize(QtCore.QSize(self.w/3,self.h/2))

        #layout = QtGui.QVBoxLayout(self)
        #layout.addWidget(self.ibtn)

        self.setGeometry(self.w / 4,self.h /7.5,self.w,self.h)
        self.setWindowTitle('Facial Imposer')
        self.setWindowIcon(QtGui.QIcon('linustechtips.jpg'))        
        self.show()
        
        
    def handleButton(self):
        self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '/')
        #self.image = QtGui.QImage(QtGui.QImageReader(self.filename).read())
        #self.ibtn.setIcon(QtGui.QIcon(QtGui.QPixmap(self.image)))
        self.changeImage(self.filename)
        #print(filename)
 
    def close_handler(self, index):
        print("close_handler called, index = {0}".format(index))
        self.notebook.removeTab(index)

    def on_pushButton_clicked(self):
        self.dialogTextBrowser.exec_()

    def changeImage(self, pathToImage):
        pixmap = QtGui.QPixmap(pathToImage)
        self.editor.setPixmap(pixmap)
        self.editor.show()
        f = open('flocation.txt', 'w')
        f.write(pathToImage)
        f.close()
        print("Hi")

    def starr(self, filename):
        run.generate(filename)
        self.renderImage()

    def renderImage(self):
        self.generated_image = QtGui.QLabel() 
        self.generated_pixmap = QtGui.QPixmap("generated.png")
        self.generated_image.setPixmap(self.generated_pixmap)
        self.generated_image.show()

def main():
    app = QtGui.QApplication(sys.argv)
    tabs = QtGui.QTabWidget()
    w = Widget(app)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
