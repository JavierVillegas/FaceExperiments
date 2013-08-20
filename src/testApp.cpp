#include "testApp.h"

#define Nx 640
#define Ny 480

/** Global variables */
//cv::String face_cascade_name = ;
//cv::String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
//cv::CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
cv::RNG rng(12345);
cv::Mat TheOutPut;
cv::Mat ForWarping;
cv::Rect FirstFace;
const int Rx =600;
const int Ry =600;
cv::Mat map_x, map_y;
//--------------------------------------------------------------
void testApp::setup(){
    
    vidGrabber.setVerbose(true);
    vidGrabber.initGrabber(Nx,Ny);
    colorImg.allocate(Nx,Ny);
    
	grayImage.allocate(Nx,Ny);
    TheOutPut = cv::Mat::zeros(Nx, Ny, CV_8UC3);
    ForWarping = cv::Mat::zeros(Nx, Ny, CV_8UC3);
    //-- 1. Load the cascades
    if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_frontalface_alt_tree.xml" ) ){
// if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_mcs_eyepair_big.xml" ) ){
//  if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_mcs_nose.xml" ) ){
    
        cout<<"--(!)Error loading cara \n"<<endl;
        };
//    if( !eyes_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_eye_tree_eyeglasses.xml" ) ){
//        cout<<"--(!)Error loading eyes\n"<<endl;
//        };
   
    
    // the nonlinear mapping
    // initializing the Mats
    map_x.create( cv::Size(Rx,Ry), CV_32FC1 );
    map_y.create( cv::Size(Rx,Ry), CV_32FC1 );
    // setting the non linear mapping
    
    float x,y;
    for (int k=0;k<Rx;k++){
		for (int q=0;q<Ry;q++){
			x=k-Rx/2.0;
			y=(q-Ry/2.0)/(Ry/2.0)*(1.7)*(CV_PI)/2.0;
            if((y>(CV_PI)/2.0)||(y<-(CV_PI)/2.0)){
                map_x.at<float>(q,k) = 10 *Rx;
            }
            else{
                map_x.at<float>(q,k) = x/cos(y)+Rx/2.0;
			}
                map_y.at<float>(q,k) = q;
		}
	}
    
}

//--------------------------------------------------------------
void testApp::update(){
    bool bNewFrame = false;
    
    
    vidGrabber.update();
    bNewFrame = vidGrabber.isFrameNew();
    
	if (bNewFrame){
        
        
        colorImg.setFromPixels(vidGrabber.getPixels(), Nx,Ny);
        cv::Mat tempoMat;
        
        tempoMat= colorImg.getCvImage();
        ForWarping = tempoMat.clone();
        TheOutPut = detectAndDisplay(tempoMat);
        
        //grayImage = colorImg;
         
    }
    
}




//--------------------------------------------------------------
void testApp::draw(){
	ofSetHexColor(0xffffff);
    
     ofxCvColorImage TempoRGB;
     TempoRGB.allocate(Nx, Ny);
     TempoRGB = TheOutPut.data;
     TempoRGB.draw(0, 0);
    vector<cv::Point2f> scene_corners(4);
    scene_corners[0] = cvPoint(FirstFace.x,FirstFace.y);
    scene_corners[1] = cvPoint(FirstFace.x + FirstFace.width,FirstFace.y);
    scene_corners[2] =cvPoint(FirstFace.x + FirstFace.width,FirstFace.y + FirstFace.height);
    scene_corners[3] =cvPoint(FirstFace.x,FirstFace.y+FirstFace.height);
    
    vector<cv::Point2f> Rewarp_corners(4);
    Rewarp_corners[0] = cvPoint(1*Rx/8.0,1*Ry/5.0);
    Rewarp_corners[1] = cvPoint(7*Rx/8.0,1*Ry/5.0);
    Rewarp_corners[2] = cvPoint(7*Rx/8.0,4*Ry/5.0);
    Rewarp_corners[3] = cvPoint(1*Rx/8.0,4*Ry/5.0);
    
    cv::Mat newH =findHomography( scene_corners, Rewarp_corners, CV_RANSAC );
    
    // the source image
    
    cv::Mat WarpDestiny;
    cv::warpPerspective(ForWarping, WarpDestiny, newH, cvSize(Rx, Ry));
    
    // converting to grayscale
    cv::Mat EdgeFace;
     cv::cvtColor( WarpDestiny, EdgeFace, CV_BGR2GRAY );
    cv::GaussianBlur(EdgeFace, EdgeFace, cv::Point(5,5), 0,0);
    Canny( EdgeFace, EdgeFace, 20, 40, 3 );
    
    EdgeFace =255-EdgeFace;
    
    ofxCvGrayscaleImage TempoGray;
    TempoGray.allocate(EdgeFace.cols, EdgeFace.rows);
    TempoGray = EdgeFace.data;
    TempoGray.draw(TempoRGB.width, 0);
  }

/** @function detectAndDisplay */
cv::Mat testApp::detectAndDisplay( cv::Mat frame )
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    
    cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(60, 60) );
    
    //-- Detect faces
//    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, cv::Size(30, 30));
    
//    cout<<faces.size()<<endl;
    for( int i = 0; i < faces.size(); i++ )
    {
//        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
//        cv::ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
        // Saving th face number 1 for recentering.
        if (i==0){
            FirstFace = faces[0];
        }
    }
    // returning the image
    return  frame;

}








//--------------------------------------------------------------
void testApp::keyPressed(int key){
    
    switch (key) {
        case 'm':
            

            break;
            
        case 's':
      
            break;
        case OF_KEY_RIGHT:
            break;
        case OF_KEY_LEFT:
            
            break;
        case 'e':
            break;
            
        case 'g':
            break;
        case 'b':
            break;
        default:
            break;
    }
    
    
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y){
    
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 
    
}