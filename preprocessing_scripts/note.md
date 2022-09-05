Use this 
```
rm -rf frames; mkdir frames; ffmpeg -i full_video_360p0010.mp4 -r 8/1 -qscale:v 4 -vf scale="320:-1" frames/test%03d.jpg
```