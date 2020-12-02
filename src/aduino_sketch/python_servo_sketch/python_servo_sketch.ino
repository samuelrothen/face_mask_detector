//Import and Variable Definition
#include <Servo.h>
Servo servoX;
int x = 90;
char input = "";

//Startup-Setup
void setup() {
  Serial.begin(9600);
  servoX.attach(2);
  servoX.write(x);
  delay(1000);
}

//Main Loop
void loop() {

 if(Serial.available()){   //Checks for Serial Data
  
  input = Serial.read();   //Stores Serial Input in Variable
  
  if(input == 'R'){      
   servoX.write(x + 1);    //Moves Servo to the Right if Input is 'R'
   x += 1;                 //Update the value of the Servo angle
  }
  
  else if(input == 'L'){ 
   servoX.write(x - 1);    //Moves Servo to the Left if Input is 'L'
   x -= 1;                 //Update the value of the Servo angle
  }
  
  else{
   servoX.write(x);
  } 
  input = "";              //Clears the Input-Variable
 
 }
 
}
