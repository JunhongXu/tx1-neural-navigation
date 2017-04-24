#include <Servo.h>
#include <ros.h>
#include <ros/time.h>
#include <std_msgs/Float32.h>

// sets the pin for trigger
const int trigPin = 13;
// sets the pin for echo
const int echoPin = 11;

// instantiate the node handle
ros::NodeHandle nh;

// instantiate publisher
std_msgs::Float32 distance;
ros::Publisher distancePub("/distance", &distance);

long duration;
float szTest;
int scanTimes=0;
int staticscantime=0;
int realscan[] ={};
int incoming[2];

void setup() {
  // init node
  nh.initNode();
  nh.advertise(distancePub);
  
  // setup pin
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
  // Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
   // delay(5);
   float d = calculateDistance();
   distance.data = d;
   distancePub.publish(&distance);
   nh.spinOnce();
}
int calculateDistance(){ 
  
  digitalWrite(trigPin, LOW); 
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH); 
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH); // Reads the echoPin, returns the sound wave travel time in microseconds
  int distance= duration*0.034/2;
  return distance;
}
