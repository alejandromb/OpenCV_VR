import numpy as np
import cv2
import time
import os
import os.path


class VideoRecorder2(): 

    # Define the codec and create VideoWriter object
    def __init__(self,filename):
        self.open=True
        #self.cap=cv2.VideoCapture(filename)
        self.cap=cv2.VideoCapture(0)    
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename+".mp4",self.fourcc, 30, (640,480))
        self.capture_time=30
        self.start_time=time.time()
        self.image_coordinates=[(0,640),(0,480)]
        self.selected_ROI=False
      
 

    def setcaptime(self,captime):
        self.capture_time=captime


   #record capture function
    def record(self):
        
        while(self.cap.isOpened() and int(time.time()-self.start_time)<self.capture_time):
            ret, frame = self.cap.read()
            
            if ret==True:
                frame = cv2.flip(frame,180)

                 # write the flipped frame
                self.out.write(frame)
                

                cv2.imshow('frame',frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
                else:
                     pass
            else:
                print("exit")
                break    

    # Release and save everything if job is finished
    def stop(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    

    # display capture function
    def display(self):
          
        while(self.cap.isOpened() and int(time.time()-self.start_time)<self.capture_time):
            ret, frame = self.cap.read()
            
            if ret==True:
                frame = cv2.flip(frame,180)             
                

                cv2.imshow('frame',frame)
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    self.stop()
                    break
                #else:
                 #   pass
            else:
                print("exit")
                break   

    def display_cut(self):
         

          
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            
            if ret==True:
            #rotating frame
                frame = cv2.flip(frame,180)
            #drawing a rectangle  
                #frame= cv2.rectangle(frame, (0, 0), (100, 100), (255,0,0), 2)   
                

                #cv2.selectROI(frame)
                
                cv2.setMouseCallback('frame', self.extract_coordinates)   
                cv2.imshow('frame',frame)
                     
                

                
                if self.selected_ROI:
                    x1 = self.image_coordinates[0][0]
                    y1 = self.image_coordinates[0][1]
                    x2 = self.image_coordinates[1][0]
                    y2 = self.image_coordinates[1][1]

                    ROIframe=frame[y1:y2,x1:x2]
                    print(x1,x2,y1,y2)
                    
                    cv2.imshow('frame2',ROIframe)
                

                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    self.stop()
                    break
                #else:
                 #   pass
            else:
                print("exit")
                break  
           
    def record_ifcontour(self):
        previous_frame=None
        flag=0
        
          
        while(self.cap.isOpened() and int(time.time()-self.start_time)<self.capture_time):

            ret, frame = self.cap.read()
            
           
         
            if ret==True:
            #rotating frame
                frame = cv2.flip(frame,180)

                x1 = 0
                x2 = 105
                y1 = 0
                y2 = 421

                ROIframe=frame[y1:y2,x1:x2] 

                gray_frame=cv2.cvtColor(ROIframe,cv2.COLOR_BGR2GRAY)
                gray_frame=cv2.GaussianBlur(gray_frame,(25,25),0)
                if previous_frame is None:
                    previous_frame=gray_frame
                    continue


                #Calculating the difference and image thresholding
              

                delta=cv2.absdiff(previous_frame,gray_frame)
                previous_frame=gray_frame
                threshold=cv2.threshold(delta,35,255, cv2.THRESH_BINARY)[1]
                #threshold=cv2.adaptiveThreshold(delta,300,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                
                # Finding all the contours
                (contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Drawing rectangles bounding the contours (whose area is > 5000)
               
                for contour in contours:

                    if cv2.contourArea(contour)>5000:
                        
                        
                        self.out.write(frame)
                        
                        flag=1                      
                                  

                    else:
                        
                        pass


                

                    (x, y, w, h)=cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)
                
                #keep recording for a 10 seconds more 30fps
                if flag>0 and flag < 300:
                    self.out.write(frame)
                    flag=flag+1
                    print("wiritg frame",flag)

                else:
                    flag=0
                
                


                #cv2.imshow("gray_frame Frame",gray_frame)
                cv2.imshow("Delta Frame",delta)
                cv2.imshow("Threshold Frame",threshold)
                cv2.imshow("Color Frame",frame)
                cv2.imshow("ROI Frame",ROIframe)

            
              

                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    self.stop()
                    break
                #else:
                 #   pass
            else:
                print("exit")
                break
    def extract_coordinates(self, event, x, y, flags, parameters):
            # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_ROI = False
            self.image_coordinates = [(x,y)]
            self.extract = True
            print("button down")

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False
            print("button up")

            self.selected_ROI = True

                # Draw rectangle around ROI
            #cv2.rectangle(, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            
            self.selected_ROI = False    

    def find_in_file(self):
                pass

    
    def video_path(self):        
        
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith(".mp4")]:
                path=os.path.join(dirpath, filename)
                
    def analyze_video(self):
        #print(path)
        baseline_image=None
        ret, first_frame = self.cap.read()
        cv2.setMouseCallback('frame', self.extract_coordinates)     

        while (self.selected_ROI== False):
            cv2.imshow('select_frame',first_frame)
            cv2.setMouseCallback('select_frame', self.extract_coordinates)              
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyWindow("select_frame")
                break

        if self.selected_ROI:
            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]
        
        
        while(self.cap.isOpened() and self.selected_ROI):
            ret, frame = self.cap.read()
            
            if ret==True:

            #rotating frame
                frame = cv2.flip(frame,180)
            #drawing a rectangle  
                #frame= cv2.rectangle(frame, (0, 0), (100, 100), (255,0,0), 2)   
                

                #cv2.selectROI(frame)           
                ROIframe=frame[y1:y2,x1:x2]
                
                    
                cv2.imshow('ROI_frame',ROIframe)
                time.sleep(0.0333)
                

                
                gray_frame=cv2.cvtColor(ROIframe,cv2.COLOR_BGR2GRAY)
                gray_frame=cv2.GaussianBlur(gray_frame,(25,25),0)
                if baseline_image is None:
                    baseline_image=gray_frame
                    continue


                #Calculating the difference and image thresholding
              

                delta=cv2.absdiff(baseline_image,gray_frame)
                threshold=cv2.threshold(delta,35,255, cv2.THRESH_BINARY)[1]
                
                # Finding all the contours
                (contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Drawing rectangles bounding the contours (whose area is > 5000)
                for contour in contours:

                    if cv2.contourArea(contour) < 15000:
                        pass
                    else:
                        print("writer record")
                        self.out.write(frame)
                    (x, y, w, h)=cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)


                cv2.imshow("gray_frame Frame",gray_frame)
                cv2.imshow("Delta Frame",delta)
                cv2.imshow("Threshold Frame",threshold)
                cv2.imshow("Color Frame",frame)

                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    self.stop()
                    break
                #else:
                 #   pass
            else:
                print("exit")
                break  


#VR1=VideoRecorder2("test1")
#VR1.display_cut()
#VR1.record_ifcontour()



