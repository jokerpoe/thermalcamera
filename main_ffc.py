#!/usr/bin/python3
import wx
import numpy
import cv2 as cv
import time
from datetime import datetime
import os, sys
from imutils import paths
import pickle
from glob import glob
from collections import namedtuple
import dlib
import face_recognition
from numpy.linalg import norm
from define import *
from imutils import face_utils
import math
import imutils
import os
import numpy as np
import os
import socket
import pygame
import RPi.GPIO as GPIO
###############################FLIR MODULES#######################
import sys
from uvctypesParabilis_v2 import *
from multiprocessing  import Queue

#*****************************
#           GUI
#*****************************
from gui import *
#*****************************
#        Application
#*****************************
import General_Globals as PG
#**************************
#     INFOMATION
#**************************

#**************************
# Add all paths in folder
Path = sys._getframe().f_code.co_filename
Path = os.path.split(Path)[0]
if not (Path in sys.path):
    sys.path.append(Path)
data = pickle.loads(open("encodings.pickle","rb").read())
width_screen = 1440
height_screen= 900
#*****************************
#        CLASS FOR GUI
#*****************************
class face_recognition_app(wx.App):
    """docstring for face recognition application"""
    def __init__(self, parent):
        super(face_recognition_app, self).__init__(parent)
        self.parent = parent
    def MainLoop(self):
        self.My_EventLoop = wx.GUIEventLoop()
        old = wx.EventLoop.GetActive ()
        wx.EventLoop.SetActive ( self.My_EventLoop )

        #***************************
        # Main Loop
        #***************************
        process_this_frame = True

        PG.App_Running = True
        Previous_Time = time.time()
        #Device context
        while PG.App_Running:
            #time.sleep ( 0.01) 
                
            ret, frame = PG.cam.read()
            frame = cv.flip(frame, 0)
            ret2, data = PG.cam2.read()
            data = cv.flip(data, 0)
            data2 = data
            data = cv.resize(data[:,:], (640, 480))
            min_c = ctok(7)
            max_c = ctok(20)
            data[0][0] = min_c
            data[-1][-1] = max_c
            frame2 = cv.LUT(raw_to_8bit(data2), generate_colour_map())
            frame2 = cv.resize(frame2[:,:], (640, 480))
            PG.data = data
            
            
            #cv.imshow("camera", cv.resize(frame2, (640, 480)))
            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            frame2 = cv.cvtColor(frame2,cv.COLOR_BGR2RGB)
            #view on interface
            try:
                #frame = Detect_masked_face(frame)
                face_recognition_name(frame,frame2)
            except Exception as e:
                print(e)
            PG.view  = True
            PG.Main_Frame.m_panel_video.Refresh()
            PG.Main_Frame.m_panel_video_thermography.Refresh()       
            process_this_frame = not process_this_frame
                      
            #bug in macos
            while self.My_EventLoop.Pending(): 
                self.My_EventLoop.Dispatch()
            #self.ProcessIdle()
            self.My_EventLoop.ProcessIdle()
            # normally 50 fps           
            sec = time.time() - Previous_Time
            fps = 1 / (sec)
            str_fps = 'FPS: %2.3f' % fps
            str_fps = str_fps +"    IP: " + PG.IP_addr
            PG.Main_Frame.SetStatusText(str_fps,0)
            Previous_Time = time.time()
        PG.cam.release()
        PG.cam2.release()
      
        #cv.destroyAllWindows()
class face_recognition_frame(frame_main1):
    """docstring for face_recognition_frame"""
    def __init__(self, parent):
        super(face_recognition_frame, self).__init__(parent)
        self.parent = parent
        self.SetIcon(wx.Icon("gui-folder/fr-arv.ico"))
        #self.m_textCtrl_debug.write("Hello !\n")
    def frame_mainOnClose(self,event):
        #Check password
        view_login = dialog_login(None)
        view_login.ShowModal()
        if PG.login :
            PG.App_Running = False
        PG.login = False 
    def m_button_exitOnButtonClick( self, event ):
        #Check password
        view_login = dialog_login(None)
        view_login.ShowModal()
        if PG.login :
            PG.App_Running = False
        PG.login = False
    def m_panel_videoOnEraseBackground( self, event ):
        pass
           
    def m_panel_videoOnPaint(self,event):
        if PG.view:
            dc = wx.BufferedPaintDC(self.m_panel_video)
            PG.frame2 = cv.resize(PG.frame2,(800,600))
            h = PG.frame2.shape[0]
            w = PG.frame2.shape[1]
            self.m_panel_video.SetSize(width =w,height=h)
            image2 = wx.Bitmap.FromBuffer(w, h, PG.frame2)   
            dc.DrawBitmap(image2, 0, 0)
            if len(PG.list_image_recognition) == 0:
                return
    def m_panel_video_thermographyOnEraseBackground( self, event ):
        pass
           
    def m_panel_video_thermographyOnPaint(self,event):
        if PG.view:
            dc = wx.BufferedPaintDC(self.m_panel_video_thermography)
            PG.frame = cv.resize(PG.frame,(800,600))
            h = PG.frame.shape[0]
            w = PG.frame.shape[1]
            self.m_panel_video_thermography.SetSize(width =w,height=h)
            image = wx.Bitmap.FromBuffer(w, h, PG.frame)
            dc.DrawBitmap(image, 0, 0)
            if len(PG.list_image_recognition) == 0:
                return
    def m_button_settingOnButtonClick( self, event ):
        view_login = dialog_login(None)
        view_login.ShowModal()
        if PG.login :
            if(PG.setting == 0):
                view_setting = dialog_setting(None)
                view_setting.Show()
        PG.login = False

def face_recognition_name(frame, frame2):

    net = PG.net
    ln = PG.ln
    COLORS = PG.COLORS
    (H, W) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (192, 192),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.6:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the frame
            #color = [int(c) for c in COLORS[1]

            if classIDs[i] == 1 and confidences[i] >= 0.5:
                #text = "{}".format("Masks")
                #cv.putText(frame, text, (x + w, y + h ),cv.FONT_ITALIC  , 1.2, [255,0,0], 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), [255,255,255], 2)
                cv.rectangle(frame, (int(x+w/2) - 20 - int(PG.values5), int(y+h/4) -20 - int(PG.values5)), (int(x+w/2) + 20  + int(PG.values5), int(y+h/4) + 20 + int(PG.values5)), [255,255,255], 2)
                #********************
                get_temp(frame, frame2, x ,y ,w ,h)
                
                #PG.frame_cut = PG.data[top1:bottom1,left1:right1]
            else:
                #cv.rectangle(frame, (x, y), (x + w, y + h), [0,0,255], 2)
                #face_recognition_name(frame, frame2
                cv.rectangle(frame, (x, y), (x + w, y + h), [255,255,255], 2)
                cv.rectangle(frame, (int(x+w/2) - 20 - int(PG.values5), int(y+h/4) -20 - int(PG.values5)), (int(x+w/2) + 20  + int(PG.values5), int(y+h/4) + 20 + int(PG.values5)), [255,255,255], 2)
                #********************
                get_temp(frame, frame2, x ,y ,w ,h)

    PG.frame = frame
    PG.frame2 = frame2
def get_temp(frame, frame2, x ,y ,w ,h):
    left1 = int(int(x+w/2) - 30 -int(PG.values4)-int(PG.values5))
    top1 = int(int(y+h/4) -20 - int(PG.values5) -int(PG.values3)-int(PG.values5))
    right1 = int(int(x+w/2) + 10 -int(PG.values4)+int(PG.values5))
    bottom1 = int(int(y+h/4) + 20 -int(PG.values3)+int(PG.values5))
    if ((bottom1-top1) <= 10) :
        PG.values5 = int(PG.values5) + 1
    #cv.rectangle(frame2, (x-13, y-13), (x + w-13, y + h-13), [255,255,255], 2)
    cv.rectangle(frame2, (int(x-13-int(PG.values4)),int(y-13-int(PG.values3))),(int(x + w-13-int(PG.values4)),int(y + h-13-int(PG.values3))), (255,255,255), 2)
    cv.rectangle(frame2, (left1,top1),(right1,bottom1), (255,255,255), 2)
    #***************************************
    PG.frame_cut = PG.data[top1:bottom1,left1:right1]
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(PG.frame_cut)
    val = ktoc(maxVal)
    a = float(format(val))
    if (a > float(PG.values0)):   
        cv.putText(frame,("{0:.1f}C".format(val)) + " FEVER",(int(x),int(y-20)), cv.FONT_HERSHEY_SIMPLEX,2.1,(255,0,0),2)
        play_beep()
    else:
        cv.putText(frame,"{0:.1f}C".format(val),(int(x),int(y-20)), cv.FONT_HERSHEY_SIMPLEX,1.8,(255,255,255),2)
def face_recognition_name_old(frame, frame2):

    net = PG.net
    ln = PG.ln
    COLORS = PG.COLORS
    (H, W) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (192, 192),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.6:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the frame
            #color = [int(c) for c in COLORS[1]

            if classIDs[i] == 1 and confidences[i] >= 0.5:
                text = "{}".format("Masks")
                cv.putText(frame, text, (x + w, y + h ),cv.FONT_ITALIC  , 1.2, [255,0,0], 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), [255,255,255], 2)
                cv.rectangle(frame, (int(x+w/2) - 20 - int(PG.values5), int(y+h/4) -20 - int(PG.values5)), (int(x+w/2) + 20  + int(PG.values5), int(y+h/4) + 20 + int(PG.values5)), [255,255,255], 2)
                #********************
                cv.rectangle(frame2, (x-13, y-13), (x + w-13, y + h-13), [255,255,255], 2)
                cv.rectangle(frame2, (int(x+w/2) - 30 - int(PG.values5), int(y+h/4) -20 - int(PG.values5)), (int(x+w/2) + 10  + int(PG.values5), int(y+h/4) + 20 + int(PG.values5)), [255,255,255], 2)
                
                #PG.frame_cut = PG.data[top1:bottom1,left1:right1]
            else:
                #cv.rectangle(frame, (x, y), (x + w, y + h), [0,0,255], 2)
                #face_recognition_name(frame, frame2)

                gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                rects = PG.face_detector(gray,0)
                for rect in rects:
                    x1 = rect.left()
                    y1 = rect.top()
                    x2 = rect.right()
                    y2 = rect.bottom()
                    shape = PG.dlib_81facelandmarks(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    a = shape[71][0]
                    b = shape[19][1]
                    
                    cv.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (255,255,255), 2)
                    cv.rectangle(frame, (int(a-20-int(PG.values5)),int(b-20-int(PG.values5))),(int(a+20+int(PG.values5)),int(b+20+int(PG.values5))), (255,255,255), 2)
                    #***************************************
                    left1 = int(a-33-int(PG.values4)-int(PG.values5))
                    top1 = int(b - 33-int(PG.values3)-int(PG.values5))
                    right1 = int(a + 7-int(PG.values4)+int(PG.values5))
                    bottom1 = int(b + 7-int(PG.values3)+int(PG.values5))
                    if ((bottom1-top1) <= 10) :
                        PG.values5 = int(PG.values5) + 1
                    cv.rectangle(frame2, (int(x1-13-int(PG.values4)),int(y1-13-int(PG.values3))),(int(x2-13-int(PG.values4)),int(y2-13-int(PG.values3))), (255,255,255), 2)
                    cv.rectangle(frame2, (left1,top1),(right1,bottom1), (255,255,255), 2)
                    #***************************************
                    PG.frame_cut = PG.data[top1:bottom1,left1:right1]
                    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(PG.frame_cut)
                    val = ktoc(maxVal)
                    a = float(format(val))
                    if (a > float(PG.values0)):
                        
                        cv.putText(frame,("{0:.1f}C".format(val)) + " FEVER",(int(x1),int(y1-20)), cv.FONT_HERSHEY_SIMPLEX,2.1,(255,0,0),2)
                        play_beep()
                    else:
                        cv.putText(frame,"{0:.1f}C".format(val),(int(x1),int(y1-20)), cv.FONT_HERSHEY_SIMPLEX,1.8,(255,255,255),2)
    PG.frame = frame
    PG.frame2 = frame2

def Detect_masked_face(frame):
    net = PG.net
    ln = PG.ln
    COLORS = PG.COLORS
    (H, W) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (192, 192),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.6:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the frame
            #color = [int(c) for c in COLORS[1]

            if classIDs[i] == 1 and confidences[i] >= 0.5:
                text = "{}".format("Masks")
                cv.putText(frame, text, (x + w, y + h ),cv.FONT_ITALIC  , 0.6, [255,0,0], 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), [0,0,255], 2)
                
                cv.rectangle(frame, (int(x/2-10), int(y/3)), (x + 20, y + 20), [0,0,255], 2)
            else:
                #cv.rectangle(frame, (x, y), (x + w, y + h), [0,0,255], 2)
                face_recognition_name(frame, frame2)
    return frame
#********change1*******
class dialog_login(MyDialog_Login):
    """docstring for dialog_login"""
    def __init__(self, parent):
        super(dialog_login, self).__init__(parent)
        self.parent = parent
        self.SetPosition(wx.Point(10,10))
        
    def m_button_cancelOnButtonClick( self, event ):
        self.Destroy()
    def m_button_confirmOnButtonClick( self, event ):
        if self.m_textCtrl_password.GetValue() == "asti":
            PG.login = True
            self.Destroy()
        else:
            PG.login = False
            answer = wx.MessageBox("Wrong password ?", "ERROR",wx.OK_DEFAULT, self)
            self.m_textCtrl_password.SetValue("")
    def MyDialog_LoginOnClose( self, event ):
        self.Destroy()
    def Enter_2_confirm( self, event ):
        if event.GetKeyCode() ==13:
            self.m_button_confirmOnButtonClick(event=None)
        else:
            event.Skip()        
class dialog_setting(MyDialog_setting):
    """docstring for dialog_login"""
    def __init__(self, parent):
        super(dialog_setting, self).__init__(parent)
        self.parent = parent
        self.SetPosition(wx.Point(1700,300))
        self.set = 0
        PG.setting = 1
        self.m_textCtrl_fever.SetValue(PG.values0)
        self.m_textCtrl_ktoc.SetValue(PG.values1)
        os.system("dbus-send --type=method_call --dest=org.onboard.Onboard /org/onboard/Onboard/Keyboard org.onboard.Onboard.Keyboard.Show")
    def MyDialog_settingOnClose( self, event ):
        if (self.set == 1):
            PG.setting = 0
            self.set = 0
            self.Destroy()
        else:
            answer3 = wx.MessageBox("Close without saving?", "Confirm",wx.OK | wx.CANCEL, self)
            if answer3 == wx.OK:
                PG.setting = 0
                self.set = 0
                getValue()
                self.Destroy()
            else:
                self.set = 0
    def m_button_closeOnButtonClick( self, event ):
        if (self.set == 1):
            PG.setting = 0
            self.set = 0
            os.system("dbus-send --type=method_call --dest=org.onboard.Onboard /org/onboard/Onboard/Keyboard org.onboard.Onboard.Keyboard.Hide")
            self.Destroy()
        else:
            answer6 = wx.MessageBox("Close without saving?", "Confirm",wx.OK | wx.CANCEL, self)
            if answer6 == wx.OK:
                PG.setting = 0
                self.set = 0
                getValue()
                os.system("dbus-send --type=method_call --dest=org.onboard.Onboard /org/onboard/Onboard/Keyboard org.onboard.Onboard.Keyboard.Hide")
                self.Destroy()
            else:
                self.set = 0

    def m_button_fever_downOnButtonClick( self, event ):
        PG.values0 = str(round(float(PG.values0) - 0.1, 2))
        self.m_textCtrl_fever.SetValue((PG.values0))

    def m_button_fever_upOnButtonClick( self, event ):
        PG.values0 = str(round(float(PG.values0) + 0.1,2) )
        self.m_textCtrl_fever.SetValue(PG.values0)

    def m_button_tempktoc_downOnButtonClick( self, event ):
        PG.values1 = str(round(float(PG.values1) - 0.1, 2))
        self.m_textCtrl_ktoc.SetValue(PG.values1)

    def m_button_tempktoc_upOnButtonClick( self, event ):
        PG.values1 = str(round(float(PG.values1) + 0.1,2))
        self.m_textCtrl_ktoc.SetValue(PG.values1)
    def m_button_mode0OnButtonClick( self, event ):
        PG.values2 = "0"

    def m_button_mode1OnButtonClick( self, event ):
        PG.values2 = "1"

    def m_button_mode2OnButtonClick( self, event ):
        PG.values2 = "2"

    def m_button_moveupOnButtonClick( self, event ):
        PG.values3 = int(PG.values3) + 1

    def m_button_moveleftOnButtonClick( self, event ):
        PG.values4 = int(PG.values4) + 1

    def m_button_moverightOnButtonClick( self, event ):
        PG.values4 = int(PG.values4) - 1

    def m_button_movebottomOnButtonClick( self, event ):
        PG.values3 = int(PG.values3) - 1

    def m_button_zoomOnButtonClick( self, event ):
        PG.values5 = int(PG.values5) + 1

    def m_button_zoomoutOnButtonClick( self, event ):
        PG.values5 = int(PG.values5) - 1

    def m_button_saveOnButtonClick( self, event ):
        answer7 = wx.MessageBox("SAVE?", "Confirm",wx.OK | wx.CANCEL, self)
        if answer7 == wx.OK:
            try:
                PG.values1 = float(self.m_textCtrl_ktoc.GetValue())
                PG.values0 = float(self.m_textCtrl_fever.GetValue())
            except:
                PG.error = 1
            if (PG.error == 0):
                PG.values0 = self.m_textCtrl_fever.GetValue() #fevertemp
                PG.values1 = self.m_textCtrl_ktoc.GetValue() #valuektoc
                PG.values2 = PG.values2 #color_mode
                PG.values3 = str(PG.values3) #move_leftright
                PG.values4 = str(PG.values4) #move_topbottom
                PG.values5 = str(PG.values5) #zoom
                saveValue()
                self.set = 1
            else:
                answer2 = wx.MessageBox("You need to fill in Value!", "ERROR", wx.OK, self)
                PG.error = 0

    def m_button_resetOnButtonClick( self, event ):
        answer4 = wx.MessageBox("Reset setting?", "Confirm",wx.OK | wx.CANCEL, self)
        if answer4 == wx.OK:
            PG.setting = 0
            self.set = 0
            getValueReset()
            time.sleep(0.5)
            saveValue()
            time.sleep(0.5)
            getValue()
            self.Destroy()
        else:
            self.set = 0   
#********change1*******
def getValue():
    f1 = open("setting","r")
    setting = f1.readlines()
    PG.values0 = setting[0].strip('\n') #fevertemp
    PG.values1 = setting[1].strip('\n') #valuektoc
    PG.values2 = setting[2].strip('\n') #color_mode
    PG.values3 = setting[3].strip('\n') #move_leftright
    PG.values4 = setting[4].strip('\n') #move_topbottom
    PG.values5 = setting[5].strip('\n') #zoom
    f1.close()
def saveValue():
    f1 = open("setting","w")
    f1.write(PG.values0 + "\n")
    f1.write(PG.values1 + "\n")
    f1.write(PG.values2 + "\n")
    f1.write(PG.values3 + "\n")
    f1.write(PG.values4 + "\n")
    f1.write(PG.values5 + "\n")
    f1.close()
def getValueReset():
    f2 = open("reset","r")
    reset = f2.readlines()
    PG.values0 = reset[0].strip('\n') #fevertemp
    PG.values1 = reset[1].strip('\n') #valuektoc
    PG.values2 = reset[2].strip('\n') #color_mode
    PG.values3 = reset[3].strip('\n') #move_leftright
    PG.values4 = reset[4].strip('\n') #move_topbottom
    PG.values5 = reset[5].strip('\n') #zoom
    f2.close()
#********************** 
def generate_colour_map():
    colorMapType = int(PG.values2) #change
    """
    Conversion of the colour map from GetThermal to a numpy LUT:
        https://github.com/groupgets/GetThermal/blob/bb467924750a686cc3930f7e3a253818b755a2c0/src/dataformatter.cpp#L6
    """

    lut = numpy.zeros((256, 1, 3), dtype=numpy.uint8)

    #colorMaps 
    colormap_grayscale = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255];

    colormap_rainbow = [1, 3, 74, 0, 3, 74, 0, 3, 75, 0, 3, 75, 0, 3, 76, 0, 3, 76, 0, 3, 77, 0, 3, 79, 0, 3, 82, 0, 5, 85, 0, 7, 88, 0, 10, 91, 0, 14, 94, 0, 19, 98, 0, 22, 100, 0, 25, 103, 0, 28, 106, 0, 32, 109, 0, 35, 112, 0, 38, 116, 0, 40, 119, 0, 42, 123, 0, 45, 128, 0, 49, 133, 0, 50, 134, 0, 51, 136, 0, 52, 137, 0, 53, 139, 0, 54, 142, 0, 55, 144, 0, 56, 145, 0, 58, 149, 0, 61, 154, 0, 63, 156, 0, 65, 159, 0, 66, 161, 0, 68, 164, 0, 69, 167, 0, 71, 170, 0, 73, 174, 0, 75, 179, 0, 76, 181, 0, 78, 184, 0, 79, 187, 0, 80, 188, 0, 81, 190, 0, 84, 194, 0, 87, 198, 0, 88, 200, 0, 90, 203, 0, 92, 205, 0, 94, 207, 0, 94, 208, 0, 95, 209, 0, 96, 210, 0, 97, 211, 0, 99, 214, 0, 102, 217, 0, 103, 218, 0, 104, 219, 0, 105, 220, 0, 107, 221, 0, 109, 223, 0, 111, 223, 0, 113, 223, 0, 115, 222, 0, 117, 221, 0, 118, 220, 1, 120, 219, 1, 122, 217, 2, 124, 216, 2, 126, 214, 3, 129, 212, 3, 131, 207, 4, 132, 205, 4, 133, 202, 4, 134, 197, 5, 136, 192, 6, 138, 185, 7, 141, 178, 8, 142, 172, 10, 144, 166, 10, 144, 162, 11, 145, 158, 12, 146, 153, 13, 147, 149, 15, 149, 140, 17, 151, 132, 22, 153, 120, 25, 154, 115, 28, 156, 109, 34, 158, 101, 40, 160, 94, 45, 162, 86, 51, 164, 79, 59, 167, 69, 67, 171, 60, 72, 173, 54, 78, 175, 48, 83, 177, 43, 89, 179, 39, 93, 181, 35, 98, 183, 31, 105, 185, 26, 109, 187, 23, 113, 188, 21, 118, 189, 19, 123, 191, 17, 128, 193, 14, 134, 195, 12, 138, 196, 10, 142, 197, 8, 146, 198, 6, 151, 200, 5, 155, 201, 4, 160, 203, 3, 164, 204, 2, 169, 205, 2, 173, 206, 1, 175, 207, 1, 178, 207, 1, 184, 208, 0, 190, 210, 0, 193, 211, 0, 196, 212, 0, 199, 212, 0, 202, 213, 1, 207, 214, 2, 212, 215, 3, 215, 214, 3, 218, 214, 3, 220, 213, 3, 222, 213, 4, 224, 212, 4, 225, 212, 5, 226, 212, 5, 229, 211, 5, 232, 211, 6, 232, 211, 6, 233, 211, 6, 234, 210, 6, 235, 210, 7, 236, 209, 7, 237, 208, 8, 239, 206, 8, 241, 204, 9, 242, 203, 9, 244, 202, 10, 244, 201, 10, 245, 200, 10, 245, 199, 11, 246, 198, 11, 247, 197, 12, 248, 194, 13, 249, 191, 14, 250, 189, 14, 251, 187, 15, 251, 185, 16, 252, 183, 17, 252, 178, 18, 253, 174, 19, 253, 171, 19, 254, 168, 20, 254, 165, 21, 254, 164, 21, 255, 163, 22, 255, 161, 22, 255, 159, 23, 255, 157, 23, 255, 155, 24, 255, 149, 25, 255, 143, 27, 255, 139, 28, 255, 135, 30, 255, 131, 31, 255, 127, 32, 255, 118, 34, 255, 110, 36, 255, 104, 37, 255, 101, 38, 255, 99, 39, 255, 93, 40, 255, 88, 42, 254, 82, 43, 254, 77, 45, 254, 69, 47, 254, 62, 49, 253, 57, 50, 253, 53, 52, 252, 49, 53, 252, 45, 55, 251, 39, 57, 251, 33, 59, 251, 32, 60, 251, 31, 60, 251, 30, 61, 251, 29, 61, 251, 28, 62, 250, 27, 63, 250, 27, 65, 249, 26, 66, 249, 26, 68, 248, 25, 70, 248, 24, 73, 247, 24, 75, 247, 25, 77, 247, 25, 79, 247, 26, 81, 247, 32, 83, 247, 35, 85, 247, 38, 86, 247, 42, 88, 247, 46, 90, 247, 50, 92, 248, 55, 94, 248, 59, 96, 248, 64, 98, 248, 72, 101, 249, 81, 104, 249, 87, 106, 250, 93, 108, 250, 95, 109, 250, 98, 110, 250, 100, 111, 251, 101, 112, 251, 102, 113, 251, 109, 117, 252, 116, 121, 252, 121, 123, 253, 126, 126, 253, 130, 128, 254, 135, 131, 254, 139, 133, 254, 144, 136, 254, 151, 140, 255, 158, 144, 255, 163, 146, 255, 168, 149, 255, 173, 152, 255, 176, 153, 255, 178, 155, 255, 184, 160, 255, 191, 165, 255, 195, 168, 255, 199, 172, 255, 203, 175, 255, 207, 179, 255, 211, 182, 255, 216, 185, 255, 218, 190, 255, 220, 196, 255, 222, 200, 255, 225, 202, 255, 227, 204, 255, 230, 206, 255, 233, 208]

    colourmap_ironblack = [
        255, 255, 255, 253, 253, 253, 251, 251, 251, 249, 249, 249, 247, 247,
        247, 245, 245, 245, 243, 243, 243, 241, 241, 241, 239, 239, 239, 237,
        237, 237, 235, 235, 235, 233, 233, 233, 231, 231, 231, 229, 229, 229,
        227, 227, 227, 225, 225, 225, 223, 223, 223, 221, 221, 221, 219, 219,
        219, 217, 217, 217, 215, 215, 215, 213, 213, 213, 211, 211, 211, 209,
        209, 209, 207, 207, 207, 205, 205, 205, 203, 203, 203, 201, 201, 201,
        199, 199, 199, 197, 197, 197, 195, 195, 195, 193, 193, 193, 191, 191,
        191, 189, 189, 189, 187, 187, 187, 185, 185, 185, 183, 183, 183, 181,
        181, 181, 179, 179, 179, 177, 177, 177, 175, 175, 175, 173, 173, 173,
        171, 171, 171, 169, 169, 169, 167, 167, 167, 165, 165, 165, 163, 163,
        163, 161, 161, 161, 159, 159, 159, 157, 157, 157, 155, 155, 155, 153,
        153, 153, 151, 151, 151, 149, 149, 149, 147, 147, 147, 145, 145, 145,
        143, 143, 143, 141, 141, 141, 139, 139, 139, 137, 137, 137, 135, 135,
        135, 133, 133, 133, 131, 131, 131, 129, 129, 129, 126, 126, 126, 124,
        124, 124, 122, 122, 122, 120, 120, 120, 118, 118, 118, 116, 116, 116,
        114, 114, 114, 112, 112, 112, 110, 110, 110, 108, 108, 108, 106, 106,
        106, 104, 104, 104, 102, 102, 102, 100, 100, 100, 98, 98, 98, 96, 96,
        96, 94, 94, 94, 92, 92, 92, 90, 90, 90, 88, 88, 88, 86, 86, 86, 84, 84,
        84, 82, 82, 82, 80, 80, 80, 78, 78, 78, 76, 76, 76, 74, 74, 74, 72, 72,
        72, 70, 70, 70, 68, 68, 68, 66, 66, 66, 64, 64, 64, 62, 62, 62, 60, 60,
        60, 58, 58, 58, 56, 56, 56, 54, 54, 54, 52, 52, 52, 50, 50, 50, 48, 48,
        48, 46, 46, 46, 44, 44, 44, 42, 42, 42, 40, 40, 40, 38, 38, 38, 36, 36,
        36, 34, 34, 34, 32, 32, 32, 30, 30, 30, 28, 28, 28, 26, 26, 26, 24, 24,
        24, 22, 22, 22, 20, 20, 20, 18, 18, 18, 16, 16, 16, 14, 14, 14, 12, 12,
        12, 10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 9,
        2, 0, 16, 4, 0, 24, 6, 0, 31, 8, 0, 38, 10, 0, 45, 12, 0, 53, 14, 0,
        60, 17, 0, 67, 19, 0, 74, 21, 0, 82, 23, 0, 89, 25, 0, 96, 27, 0, 103,
        29, 0, 111, 31, 0, 118, 36, 0, 120, 41, 0, 121, 46, 0, 122, 51, 0, 123,
        56, 0, 124, 61, 0, 125, 66, 0, 126, 71, 0, 127, 76, 1, 128, 81, 1, 129,
        86, 1, 130, 91, 1, 131, 96, 1, 132, 101, 1, 133, 106, 1, 134, 111, 1,
        135, 116, 1, 136, 121, 1, 136, 125, 2, 137, 130, 2, 137, 135, 3, 137,
        139, 3, 138, 144, 3, 138, 149, 4, 138, 153, 4, 139, 158, 5, 139, 163,
        5, 139, 167, 5, 140, 172, 6, 140, 177, 6, 140, 181, 7, 141, 186, 7,
        141, 189, 10, 137, 191, 13, 132, 194, 16, 127, 196, 19, 121, 198, 22,
        116, 200, 25, 111, 203, 28, 106, 205, 31, 101, 207, 34, 95, 209, 37,
        90, 212, 40, 85, 214, 43, 80, 216, 46, 75, 218, 49, 69, 221, 52, 64,
        223, 55, 59, 224, 57, 49, 225, 60, 47, 226, 64, 44, 227, 67, 42, 228,
        71, 39, 229, 74, 37, 230, 78, 34, 231, 81, 32, 231, 85, 29, 232, 88,
        27, 233, 92, 24, 234, 95, 22, 235, 99, 19, 236, 102, 17, 237, 106, 14,
        238, 109, 12, 239, 112, 12, 240, 116, 12, 240, 119, 12, 241, 123, 12,
        241, 127, 12, 242, 130, 12, 242, 134, 12, 243, 138, 12, 243, 141, 13,
        244, 145, 13, 244, 149, 13, 245, 152, 13, 245, 156, 13, 246, 160, 13,
        246, 163, 13, 247, 167, 13, 247, 171, 13, 248, 175, 14, 248, 178, 15,
        249, 182, 16, 249, 185, 18, 250, 189, 19, 250, 192, 20, 251, 196, 21,
        251, 199, 22, 252, 203, 23, 252, 206, 24, 253, 210, 25, 253, 213, 27,
        254, 217, 28, 254, 220, 29, 255, 224, 30, 255, 227, 39, 255, 229, 53,
        255, 231, 67, 255, 233, 81, 255, 234, 95, 255, 236, 109, 255, 238, 123,
        255, 240, 137, 255, 242, 151, 255, 244, 165, 255, 246, 179, 255, 248,
        193, 255, 249, 207, 255, 251, 221, 255, 253, 235, 255, 255, 24]

    def chunk(ulist, step):
        return map(lambda i: ulist[i: i + step], range(0, len(ulist), step))

    if (colorMapType == 1):
        chunks = chunk(colormap_rainbow, 3)
    elif (colorMapType == 2):
        chunks = chunk(colormap_grayscale, 3)
    else:
        chunks = chunk(colourmap_ironblack, 3)

    red = []
    green = []
    blue = []

    for chunk in chunks:
        red.append(chunk[0])
        green.append(chunk[1])
        blue.append(chunk[2])

    lut[:, 0, 0] = blue

    lut[:, 0, 1] = green

    lut[:, 0, 2] = red

    return lut
#*****************************
def raw_to_8bit(data):
  cv.normalize(data, data, 0, data.max()*2, cv.NORM_MINMAX)
  numpy.right_shift(data, 2, data)
  return cv.cvtColor(numpy.uint8(data), cv.COLOR_GRAY2RGB)
  
def ktoc(val):
    val = (round(((val - 27315) / 100.0),2))
    return calib_ktoc(val) #change
    
def calib_ktoc(val):
    return val + float(PG.values1)

def Check_camera_index():
    index_camera_thermal = '1'
    index_normal_camera = '0'
    try:
        os.system('rm camera_id.txt')
    except:
        pass
    os.system('v4l2-ctl --list-devices >> camera_id.txt')

    camedid_from_file = open("camera_id.txt","r")
    camera_id = camedid_from_file.readlines()
    #print(camedid_from_file)
    print(camera_id)
    camera_id1 = camera_id[0]
    if camera_id1.find('PureThermal') == -1:
        camera_thermal = 3
        camera_webcam = 0
    else:
        camera_thermal = 0
        camera_webcam = 3

    index_normal_camera = camera_id[camera_webcam+1].replace('\t', '')
    index_normal_camera = index_normal_camera.replace('\n', '')
    index_camera_thermal = camera_id[camera_thermal+1].replace('\t', '')
    index_camera_thermal = index_camera_thermal.replace('\n', '')
    return index_camera_thermal, index_normal_camera

def Load_model():
    #sub_folder = "Masked_face"
    sub_folder = "model_face_detect"
    print('[INFO] LOADING MASKED FACE DATA')
    labelsPath = os.path.sep.join([sub_folder, "coco.names"])
    weightsPath = os.path.sep.join([sub_folder, "yolov3.weights"])
    configPath = os.path.sep.join([sub_folder, "yolov3.cfg"])
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    try:
        net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        print("[INFO] Using GPU CUDA...")
    except:
        print("[INFO] NO GPU support...")
        
    return net, ln, COLORS

def get_ip_address():
    try :
        socket.gethostbyname(socket.gethostname())
        return socket.gethostbyname(socket.gethostname())
    except:
        return 'NULL'


def play_beep():
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
          
def ctok(val):
    return (val * 100.0) + 27315
    
def startStream():
  global devh
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    #exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      print("done starting stream, displaying settings")
      print_shutter_info(devh)
      print("resetting settings to default")
      set_auto_ffc(devh)
      set_gain_high(devh)
      print("current settings")
      print_shutter_info(devh)

    except:
      #libuvc.uvc_unref_device(dev)
      print('Failed to Open Device')
  except:
    #libuvc.uvc_exit(ctx)
    print('Failed to Find Device')
    exit(1)

    
#*****************************          

#       Main program
#*****************************  
def main():
    print('==========================================')
    print(datetime.now())
    print('[INFO] Checking camera ....', end ='')
    index_Tcamera, index_Ncamera = Check_camera_index()
    print(' Index_Tcamera = ',index_Tcamera, end =' ')
    print('Index_Ncamera = ',index_Ncamera)
    PG.net, PG.ln, PG.COLORS = Load_model()
    print("[FEVER DETECTOR] STARTING PROGRAM")
    pygame.mixer.init()
    pygame.mixer.music.load("gui-folder/beep-08b.mp3")
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    getValue()
    PG.face_detector   = dlib.get_frontal_face_detector()
    PG.dlib_81facelandmarks = dlib.shape_predictor(PG.predictor_81_shape_path)
    PG.IP_addr = get_ip_address()
    time.sleep(10)
    PG.cam = cv.VideoCapture(index_Ncamera)
    time.sleep(2)
    index_Tcamera_id = index_Tcamera.replace('/dev/video','')
    i = 0
    while i <=5:
        PG.cam2 = cv.VideoCapture(int(index_Tcamera_id) + cv.CAP_V4L2)
        res, test_Tcamera = PG.cam2.read()
        if res:
            break
        else:
            i = i + 1
            PG.cam2.release()
            time.sleep(1)
    PG.cam2.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"Y16 "))
    PG.cam2.set(cv.CAP_PROP_CONVERT_RGB, 0)
    PG.cam.set(cv.CAP_PROP_BUFFERSIZE, 1)
    PG.cam.set(cv.CAP_PROP_FRAME_WIDTH,640)
    PG.cam.set(cv.CAP_PROP_FRAME_HEIGHT,480)
    time.sleep(1)
    PG.Main_App =  face_recognition_app(None)
    PG.Main_Frame = face_recognition_frame(None)
    PG.Main_Frame.Show()
    PG.Main_App.MainLoop()
#*****************************  
if __name__ == '__main__':
    main()
    '''
    try:
        main()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, fname, exc_tb.tb_lineno)'''