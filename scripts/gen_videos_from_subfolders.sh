DEFAULT_SRC_DIR='.'
if [ -n "$1" ]
then 
  echo ">> Given a source directory of folders: $1"
  echo ">> Set SRC_DIR = $1"
  SRC_DIR=$1
else
  echo ">> Have not given a source directory of folders!"
  echo ">> By default, set SRC_DIR = $DEFAULT_SRC_DIR"
  SRC_DIR=$DEFAULT_SRC_DIR
fi 

# Create a video directory
VIDEO_DIR=${SRC_DIR}/videos
TRASH_DIR=/home/tynguyen/junk
mkdir $VIDEO_DIR 
mkdir $TRASH_DIR 
echo "------------------------"
echo ">> Created video folder $VIDEO_DIR"


echo ">> Listing subfolders in the SRC_DIR:"
for d in ${SRC_DIR}/* ; do
  [ -d "$d" ] || continue 
  echo "-------------------------"
  echo "Found directory $d"
  
  # Get the subfolder name
  for i in $(echo $d | tr "/" "\n")
  do
	model_name=$i
  done
  
  # Saving directory for the video
  video_name="${model_name}.mp4"
  video_path=${VIDEO_DIR}/${video_name}
  echo ">> Video is saved to : $video_path" 
  img_files=$(find ${d}/*.png)
  #echo ">> Image files: $img_files"
  
  # Rename all images
  #x=0; 
  #for i in $img_files; 
  #do 
  #  counter=$(printf %06d $x); 
  #  ln "$i" $TRASH_DIR/img"$counter".jpg; x=$(($x+1)); 
  #  echo ">> Move image $i to tmp/img$counter.jpg" 
  #done
  
  img_count=0; for i in $img_files; do img_count=$(($img_count+1)); done  
  echo ">> There are $img_count images in the subfolder"
  if [ $img_count -lt  5 ]
  then 
    echo ">> Skipppp this model since there are not enough images!"
    echo "*******************FAIL************************"
  else
    cat $img_files | ffmpeg -y -f image2pipe -i - $video_path 
    echo ">> Complete saving video : $video_path" 
    echo "*******************SUCCESS************************"

  fi 
  
done 
