![My image](https://github.com/bemcho/home-ai-reloaded/blob/master/home-ai-demo.png)
# home-ai-reloaded
home-ai rewriten in c++
##So far you can not train with this implementation
but you can use home-ai to train nad just copy the xml file from (home-ai/resources/data/facerecognizers/ 	lbphFaceRecognizer.xm)
```
here d:\home-ai-reloaded\HomeAIReloaded\x64\Release\
or/and here d:\home-ai-reloaded\HomeAIReloaded\x64\Debug\
```
#Set up let say home-ai-reloaded is saved as: d:\home-ai-reloaded

>extract
```
        d:\home-ai-reloaded\HomeAIReloaded\x64\Release\*.7z
        d:\home-ai-reloaded\HomeAIReloaded\x64\Debug\*.7z   
```

###add d:\home-ai-reloaded\HomeAIReloaded\x64\bin 
   to your system Path(this is needed in order dll to be found during run time)
>you can use your own builds of dll and lib files of course.

##Open solution properties:
>under c++ for both Debug and Release:
```
   Additional Include Directories d:\home-ai-reloaded\HomeAIReloaded\x64\include
   Additional #using Directories d:\home-ai-reloaded\HomeAIReloaded\x64\include
```
##Under Linker-> General:
  >Additional Library d:\home-ai-reloaded\HomeAIReloaded\x64\lib
  
##Under Linker ->Input:
  >Copy and paste all *.lib files here
  >Additional Dependencies    
                For Debug Configuration:
```
                          opencv_aruco310d.lib
                          opencv_bgsegm310d.lib
                          opencv_calib3d310d.lib
                          opencv_ccalib310d.lib
                          opencv_core310d.lib
                          opencv_datasets310d.lib
                          opencv_dnn310d.lib
                          opencv_dpm310d.lib
                          opencv_face310d.lib
                          opencv_features2d310d.lib
                          opencv_flann310d.lib
                          opencv_fuzzy310d.lib
                          opencv_highgui310d.lib
                          opencv_imgcodecs310d.lib
                          opencv_imgproc310d.lib
                          opencv_ml310d.lib
                          opencv_objdetect310d.lib
                          opencv_optflow310d.lib
                          opencv_photo310d.lib
                          opencv_plot310d.lib
                          opencv_reg310d.lib
                          opencv_rgbd310d.lib
                          opencv_shape310d.lib
                          opencv_stereo310d.lib
                          opencv_stitching310d.lib
                          opencv_structured_light310d.lib
                          opencv_superres310d.lib
                          opencv_surface_matching310d.lib
                          opencv_text310d.lib
                          opencv_tracking310d.lib
                          opencv_ts310d.lib
                          opencv_video310d.lib
                          opencv_videoio310d.lib
                          opencv_videostab310d.lib
                          opencv_world310d.lib
                          opencv_ximgproc310d.lib
                          opencv_xobjdetect310d.lib
                          opencv_xphoto310d.lib
```
  For Release Configuration:
  ```
              opencv_aruco310.lib
              opencv_bgsegm310.lib
              opencv_calib3d310.lib
              opencv_ccalib310.lib
              opencv_core310.lib
              opencv_datasets310.lib
              opencv_dnn310.lib
              opencv_dpm310.lib
              opencv_face310.lib
              opencv_features2d310.lib
              opencv_flann310.lib
              opencv_fuzzy310.lib
              opencv_highgui310.lib
              opencv_imgcodecs310.lib
              opencv_imgproc310.lib
              opencv_ml310.lib
              opencv_objdetect310.lib
              opencv_optflow310.lib
              opencv_photo310.lib
              opencv_plot310.lib
              opencv_reg310.lib
              opencv_rgbd310.lib
              opencv_shape310.lib
              opencv_stereo310.lib
              opencv_stitching310.lib
              opencv_structured_light310.lib
              opencv_superres310.lib
              opencv_surface_matching310.lib
              opencv_text310.lib
              opencv_tracking310.lib
              opencv_ts310.lib
              opencv_video310.lib
              opencv_videoio310.lib
              opencv_videostab310.lib
              opencv_ximgproc310.lib
              opencv_xobjdetect310.lib
              opencv_xphoto310.lib
```
