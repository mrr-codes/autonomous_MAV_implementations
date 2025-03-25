/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 */

 #include "modules/our_avoider/our_avoider.h"
 #include "firmwares/rotorcraft/guidance/guidance_h.h"
 #include "firmwares/rotorcraft/navigation.h"
 #include "generated/airframe.h"
 #include "state.h"
 #include "modules/core/abi.h"
 #include <time.h>
 #include <stdio.h>
 #include <math.h>
 
 #define NAV_C // needed to get the nav functions like Inside...
 #include "generated/flight_plan.h"
 #include "../../boards/ardrone2.h"
 
 #define ORANGE_AVOIDER_VERBOSE TRUE
 
 #define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
 #if ORANGE_AVOIDER_VERBOSE
 #define VERBOSE_PRINT PRINT
 #else
 #define VERBOSE_PRINT(...)
 #endif
 
 int sign(int num);
 
 enum navigation_state_t {
   SAFE,
   FRONTAL_OBSTACLE,
   SEARCH_HEADING,
   OUT_OF_BOUNDS,
   TOP_LINE,
   RIGHT_LINE,
   BOTTOM_LINE,
   LEFT_LINE
 };
 
 // module settings, choosing between cnn and orange avoider
 int cnn_enabled = 1;
 
 // cyberzoo bounds
 float OUTER_BOUNDS = 3.1f;
 float INNER_BOUNDS = 2.6f;
 float SAFE_BOUNDS = 2.5f;
 // angle of cyberzoo wrt NED frame
 float anglewrtEnu = -35;
 
 // avoidance velocity settings
 float unsafe_xvel = .5f;
 float unsafe_yvel = .5f;
 float heading_turn_rate = 1.f;
 float heading_search_rate = 0.25f;
 
 // confidence of an object in front of the drone
 float cnn_w_avg = 0.f;
 // history of objects being on the right or left of the drone
 float cnn_sum_r = 0.f;
 float cnn_sum_l = 0.f;
 // last known safe coordinates with no obstacles nearby
 float spx = 0.0f;
 float spy = 0.0f;
 
 // Global settings - changable in gcs
 float slow_mode_safe_xvel = .4f;
 float slow_mode_safe_yvel = .3f;
 float fast_mode_safe_xvel = .6f;
 float fast_mode_safe_yvel = .3f;
 
 float cnn_frontal_obstacle_threshold = 0.8f;
 
 float xvel;
 float yvel;


 float cnn_weight = 0.7;
  
 
  
  
  // Define CBF parameters
 float cbf_param = 2.0f; // CBF tuning parameter
 
 // cornering velocities and turn rates when close to an edge
 float cornering_xvel = 0.1f;
 float cornering_yvel = 0.2f;
 float cornering_turn_rate = 1.0f;
 
 // yaw rate proportional factors
 float k_lat = .4f;
 float k_heading = .6f;
 
 // Orange avoider area sizes
   float region_size = 0.f; 
   float slice_size = 0.f; 
 
 
   float of = 0.f; // total orange fraction (of) in slice
   float of_a = 0.f; // of in left region
   float of_b = 0.f; // of in left-center region
   float of_c = 0.f; // of in right-center region
   float of_d = 0.f; // of in right region
 
 
 // CNN probabilites for left center and right image
 float cnn_p_left = 0.0f;
 float cnn_p_center = 0.0f;
 float cnn_p_right = 0.0f;
 
 // moving average CNN probabilites for left center and right image
 #define MOVING_AVERAGE_SIZE 5
 float cnn_n_prev_prob_left[MOVING_AVERAGE_SIZE] = {0.0f};
 float cnn_n_prev_prob_center[MOVING_AVERAGE_SIZE] = {0.0f};
 float cnn_n_prev_prob_right[MOVING_AVERAGE_SIZE] = {0.0f};
 
 // orange count in regions
 int16_t color_count_a = 0;
 int16_t color_count_b = 0; 
 int16_t color_count_c = 0; 
 int16_t color_count_d = 0; 
 
 enum navigation_state_t navigation_state = SAFE;
 
 // part of the code responsible for detecting the orange pixels in the image
 // and sorting them into the 4 parts of the image
 #ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
 #define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
 #endif
 static abi_event color_detection_ev;
 static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                                int16_t count_region_a, int16_t count_region_b,
                                int16_t count_region_c, int16_t count_region_d,
                                int32_t __attribute__((unused)) quality, int16_t __attribute__((unused)) extra)
 {
   color_count_a = count_region_a;
   color_count_b = count_region_b;
   color_count_c = count_region_c;
   color_count_d = count_region_d;
 
   of = (color_count_a + color_count_b + color_count_c + color_count_d) / slice_size;
   of_a = color_count_a / region_size;
   of_b = color_count_b / region_size; 
   of_c = color_count_c / region_size;
   of_d = color_count_d / region_size;
 }
 
 // Assumes length MOVING_AVERAGE_SIZE
 // moves the data further in time to allow for new data
 void move_data_back(float f_array[MOVING_AVERAGE_SIZE]){
   for (int i = 0; i < MOVING_AVERAGE_SIZE-1; i++){
     f_array[i+1] = f_array[i];
   }
 }
 
 // Assumes length MOVING_AVERAGE_SIZE
 // computes the weighted moving average of the data
 float array_weighted_moving_average(float f_array[MOVING_AVERAGE_SIZE]){
   float wma = 0.0f;
   for (int i = 0; i < MOVING_AVERAGE_SIZE; i++){
       wma += f_array[i]*(MOVING_AVERAGE_SIZE-i);
   }
   wma /= (MOVING_AVERAGE_SIZE*(MOVING_AVERAGE_SIZE+1)/2);
   return wma;
 }
 
 // function that calls the cnn in order to obtain the probability of an
 // object being in the left center or right part of the screen
 #ifndef CNN_OBS_ID
 #define CNN_OBS_ID ABI_BROADCAST
 #endif
 static abi_event cnn_obs_ev;
 static void cnn_obs_cb(uint8_t __attribute__((unused)) sender_id,
                        float prob_left,
                        float prob_center,
                        float prob_right)
 {
   
 
   // cnn data is first being moved in time
   // then the new data point is added
   move_data_back(cnn_n_prev_prob_left);
   cnn_n_prev_prob_left[0] = prob_left;
   
   move_data_back(cnn_n_prev_prob_center);
   cnn_n_prev_prob_center[0] = prob_center;
 
   move_data_back(cnn_n_prev_prob_right);
   cnn_n_prev_prob_right[0] = prob_right;
 
   // the weighted moving average of the data is computed
   cnn_p_left = array_weighted_moving_average(cnn_n_prev_prob_left);
   cnn_p_center = array_weighted_moving_average(cnn_n_prev_prob_center);
   cnn_p_right = array_weighted_moving_average(cnn_n_prev_prob_right);
 
  //  printf("recieved: l: %f, c: %f, r: %f", cnn_p_left, cnn_p_center, cnn_p_right);
 
 }
 
 // defining the sizes of the region and the entire slice
 void our_avoider_init(void)
 {
   srand(time(NULL));
 
   region_size = front_camera.output_size.w / 3 * front_camera.output_size.h / 4; 
   slice_size = front_camera.output_size.w / 3 * front_camera.output_size.h; 
 
   xvel = fast_mode_safe_xvel;
   yvel = fast_mode_safe_yvel;
   AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
   AbiBindMsgCNN_OBS(CNN_OBS_ID, &cnn_obs_ev, cnn_obs_cb);
 
 }
 
 
 void our_avoider_periodic(void)
 {
   // Only run the mudule if we are in the correct flight mode
   if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
     navigation_state = SAFE;
     return;
   }
 
   
   // rotating the ENU coordinate positions in order to fit the cyberzoo coordinate system
   float newx = +cos(RadOfDeg(anglewrtEnu))*stateGetPositionEnu_f() ->x - sin(RadOfDeg(anglewrtEnu))*stateGetPositionEnu_f() ->y;
   float newy = sin(RadOfDeg(anglewrtEnu))*stateGetPositionEnu_f() ->x + cos(RadOfDeg(anglewrtEnu))*stateGetPositionEnu_f() ->y;
 
   // obtaining heading angle
   float heading = stateGetNedToBodyEulers_f()->psi;
   // normalizing the heading angle and adjusting it to the cyberzoo coordinates
   float heading_deg = DegOfRad(heading) - anglewrtEnu;
   if (heading_deg > 180){
     heading_deg = heading_deg - 360;
   }
   

   switch (navigation_state){
     case SAFE:
       VERBOSE_PRINT("STATE: SAFE\n");
       // check if drone is out of bounds, if out of bounds move to state out of bounds
       if (fabsf(newx) > OUTER_BOUNDS || fabsf(newy) > OUTER_BOUNDS) {
         // if outside outer bounds, check if heading is facing out of bounds, needs to be checked for each edge
         if(newy >= OUTER_BOUNDS && ((0 <= heading_deg && heading_deg <= 90) || (-90 <= heading_deg && heading_deg <= 0))) {
           // top side
           navigation_state = OUT_OF_BOUNDS;
           break;
         } else if(newy <= -OUTER_BOUNDS && ((90 <= heading_deg  && heading_deg <= 180) || (-180 <= heading_deg && heading_deg <= -90))) {
           // bottom side 
           navigation_state = OUT_OF_BOUNDS;
           break;
         } else if(newx <= -OUTER_BOUNDS && -180 <= heading_deg && heading_deg <= 0) {
           // left side
           navigation_state = OUT_OF_BOUNDS;
           break;
         } else if(newx >= OUTER_BOUNDS && 0 <= heading_deg && heading_deg <= 180) {
           // right side
           navigation_state = OUT_OF_BOUNDS;
           break;
         }
       }
 
       // if outside inner bounds, turn inside without stopping
       if (newy >= INNER_BOUNDS && ((0 <= heading_deg && heading_deg <= 90) || (-90 <= heading_deg && heading_deg <= 0))) {
         // close to the top edge
         navigation_state = TOP_LINE;
         break;
       } else if (newx >= INNER_BOUNDS && 0 <= heading_deg && heading_deg <= 180) {
         // close to the right edge
         navigation_state = RIGHT_LINE;
         break;
       } else if (newy <= -INNER_BOUNDS&& ((90 <= heading_deg  && heading_deg <= 180) || (-180 <= heading_deg && heading_deg <= -90))) {
         // close to the bottom edge
         navigation_state = BOTTOM_LINE;
         break;
       } else if (newx <= -INNER_BOUNDS && -180 <= heading_deg && heading_deg <= 0) {
         // close to the left edge
         navigation_state = LEFT_LINE;
         break;
       }
       // if the drone is not out of bounds or close to out of bounds
       // set faster velocities if orange avoider is used
       // set slower velocities if cnn is used
       // set slow vlocities if were out of bounds
       if (fabsf(newx) < SAFE_BOUNDS && fabsf(newy) < SAFE_BOUNDS) {
         xvel = slow_mode_safe_xvel;
         yvel = slow_mode_safe_yvel;
       }
 
  // Define threshold for switching between orange avoider and CNN
  float orange_threshold = 0.6f; // Threshold for the sum of of_b and of_c

  // Check if the sum of of_b and of_c exceeds the threshold
  bool use_orange_avoider = (of_b + of_c) > orange_threshold;
  printf("OF: %f",of_b+of_c);
 if ((of_b > 0.3 && of_c > 0.3) || (of_b > 0.6f) || (of_c > 0.6f)){
  VERBOSE_PRINT("ORANGE_FRONT_OBSTACLE\n");
 }

 
 if ((cnn_p_left > cnn_frontal_obstacle_threshold && cnn_p_center > cnn_frontal_obstacle_threshold - 0.3) ||
     cnn_p_center > cnn_frontal_obstacle_threshold - 0.05 ||
     (cnn_p_right > cnn_frontal_obstacle_threshold && cnn_p_center > cnn_frontal_obstacle_threshold - 0.3) ||
     (of_b > 0.4 && of_c > 0.4) || (of_b > 0.75f) || (of_c > 0.75f)) {
   // If a frontal obstacle is detected, switch to the FRONTAL_OBSTACLE state
   navigation_state = FRONTAL_OBSTACLE;
   spx = newx; // Save the last known safe position
   spy = newy;
 }
//  float applied_forward_velocity ;
//  float applied_lateral_velocity;
//  float applied_heading_rate;

  // // Decide which system to use based on the sum of of_b and of_c
  // if (use_orange_avoider) {
  //   // Use orange avoider system
  //   applied_forward_velocity = (1 - of) * xvel; // Orange avoider forward velocity
  //   applied_lateral_velocity = (k_inner * (of_b - of_c) + k_outer * (of_a - of_d)) * yvel; // Orange avoider lateral velocity
  //   applied_heading_rate = (k_inner * (of_b - of_c) + k_outer * (of_a - of_d)) * heading_turn_rate; // Orange avoider heading rate
  //   VERBOSE_PRINT("STATE: ORANGE_AVOIDER\n");
  // } else {
  //   // Use CNN-based system
  //   applied_forward_velocity = (1 - cnn_w_avg) * xvel; // CNN forward velocity
  //   applied_lateral_velocity = 0.4 * (cnn_p_left - cnn_p_right) * yvel; // CNN lateral velocity
  //   applied_heading_rate = 0.3 * (cnn_p_left - cnn_p_right) * heading_turn_rate; // CNN heading rate
  //   VERBOSE_PRINT("STATE: CNN_AVOIDER\n");
  // }
    // Adjust the CNN weight based on orange avoider detection
    if (of > 0.4f) // Threshold for orange avoider detection
    {
      cnn_weight = 0.0f; // Reduce CNN weight if orange avoider detection is high
    }
    else
    {
      cnn_weight = 1.0f; // Default CNN weight
    }
    float orange_weight=1.0f -cnn_weight;
  


  // Compute the weighted average for CNN probabilities
  cnn_w_avg = (0.25 * cnn_p_left + 0.5 * cnn_p_center + 0.25 * cnn_p_right) / 3;
 
       // Calculate velocities and heading rates for both systems
       float or_forward_velocity = (1 - of) * xvel; // Orange avoider forward velocity
       float or_lateral_velocity = (0.4 * (of_b - of_c) + 0.6 * (of_a - of_d)) * yvel; // Orange avoider lateral velocity
       float or_heading_rate = (0.4 * (of_b - of_c) + 0.6 * (of_a - of_d)) * heading_turn_rate; // Orange avoider heading rate
   
       float ob_forward_velocity = (1 - (cnn_w_avg+0.5*cnn_p_center)) * xvel; // CNN forward velocity
       float ob_lateral_velocity = k_lat * (cnn_p_left - cnn_p_right) * yvel; // CNN lateral velocity
       float ob_heading_rate = k_heading * (cnn_p_left - cnn_p_right) * heading_turn_rate; // CNN heading rate
   
       // Combine the velocities and heading rates using weighted averages
       float applied_forward_velocity = (cnn_weight * ob_forward_velocity) + (orange_weight * or_forward_velocity);
       float applied_lateral_velocity = (cnn_weight * ob_lateral_velocity) + (orange_weight * or_lateral_velocity);
       float applied_heading_rate = (cnn_weight * ob_heading_rate) + (orange_weight * or_heading_rate);

  // CBF logic to ensure the drone stays within bounds
  float distance = sqrt(newx * newx + newy * newy);
  float h_inner = (INNER_BOUNDS-0.3f)-distance; // Barrier function for inner bound

  // Derivatives of the barrier functions
  float dh_inner_dx = -newx / distance;
  float dh_inner_dy = -newy / distance;

  // CBF conditions
  float cbf_inner = dh_inner_dx * applied_forward_velocity + dh_inner_dy * applied_lateral_velocity + cbf_param * h_inner;

  // Adjust velocities if CBF conditions are violated
  if (cbf_inner < 0) {
    float scale_factor = 0.2f; // Reduce velocity to stay within bounds
    applied_forward_velocity *= scale_factor;
    applied_lateral_velocity *= scale_factor;
    applied_forward_velocity = fmax(applied_forward_velocity, 0.1f); // Apply minimum limit
  }

  // Set the final velocities and heading rates
  guidance_h_set_body_vel(applied_forward_velocity, applied_lateral_velocity);
  guidance_h_set_heading_rate(applied_heading_rate);

  break;
 
 
     case FRONTAL_OBSTACLE:
       VERBOSE_PRINT("STATE: FRONTAL_OBSTACLE\n");
       // stop the drone
       guidance_h_set_heading_rate(0.f);
 
       // move the drone back to the lasat known safe location
       float dvx = spx - newx;
       float dvy = spy - newy;
 
       float drone_dvx = +cos(RadOfDeg(90 - heading_deg)) * dvx + sin(RadOfDeg(90 - heading_deg)) * dvy;
       float drone_dvy = -sin(RadOfDeg(90 - heading_deg)) * dvx + cos(RadOfDeg(90 - heading_deg)) * dvy;
 
       float dist = sqrt(powf(spx - newx, 2) + powf(spy - newy, 2));
 
       guidance_h_set_body_vel(0.5 * drone_dvx, -0.5 * drone_dvy);
 
       // if the distance to the last safe location is small enough search for safe heading
       // this is made to prevent the drone for hovering too close to the obstacle
       if(dist < 0.15) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
       }
       break;
 
 
     case SEARCH_HEADING:
       // first uses the cnn history to determine which side has less objects in order to search for safe heading for as short as possible
       guidance_h_set_heading_rate(sign(cnn_sum_l - cnn_sum_r) * heading_search_rate);
       VERBOSE_PRINT("STATE: SEARCH_HEADING\n");
       // if using orange avoindance assume safe heading if no obstacle in center regiouns
       if(cnn_enabled == 0) {
         if(of_b < 0.4f && of_c < 0.4f) {
           navigation_state = SAFE;
         }
       }
 
       // if using cnn avoidance assume safe heading if no obstacle in safe regoind and restart the history of obstacles counter
       if(cnn_enabled == 1) {
         if(cnn_p_center < 0.5f) {
           navigation_state = SAFE;
           cnn_sum_r = 0.f;
           cnn_sum_l = 0.f;
         }
       }
       break;
 
 
     case TOP_LINE:
       VERBOSE_PRINT("STATE: TOP_LINE\n");
       // in case we are close to the edge
       // first look at if there are obstacles and we are close to the edge in which case just stop and search for safe heading
       if(cnn_p_center > 0.8) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;  
       } else if((of_b > 0.4 && of_c > 0.4) || (of_b > 0.75f) || (of_b > 0.75f)) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;
       }
 
       // first we check if we have drifted further to the edge of the cyberzoo
       // if there are no obstacles nearby, we look at our current heading and decide to turn either left or right
       if(newy >= OUTER_BOUNDS && ((0 <= heading_deg && heading_deg <= 90) || (-90 <= heading_deg && heading_deg <= 0))) {
           navigation_state = OUT_OF_BOUNDS;
       } else if(heading_deg >= 0 && heading_deg <=110) {
         //turn right
         guidance_h_set_body_vel(cornering_xvel, cornering_yvel);
         guidance_h_set_heading_rate(cornering_turn_rate);
         navigation_state = SAFE;
       } else if (heading_deg <= 0 && heading_deg >= -110) {
         //turn left
         guidance_h_set_body_vel(cornering_xvel, -cornering_yvel);
         guidance_h_set_heading_rate(-cornering_turn_rate);
         navigation_state = SAFE;
       } else {
         guidance_h_set_body_vel(cornering_xvel, 0);
         guidance_h_set_heading_rate(RadOfDeg(0));
         navigation_state = SAFE;
       }
       break;
 
 
     case RIGHT_LINE:
       VERBOSE_PRINT("STATE: RIGHT_LINE\n");
       // in case we are close to the edge
       // first look at if there are obstacles and we are close to the edge in which case just stop and search for safe heading
       if(cnn_p_center > 0.8) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;  
       } else if((of_b > 0.4 && of_c > 0.4) || (of_b > 0.75f) || (of_b > 0.75f)) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;
       }
 
       // first we check if we have drifted further to the edge of the cyberzoo
       // if there are no obstacles nearby, we look at our current heading and decide to turn either left or right
       if(newx >= OUTER_BOUNDS && 0 <= heading_deg && heading_deg <= 180) {
           navigation_state = OUT_OF_BOUNDS;
       } else if((heading_deg >= 90 && heading_deg <=180) || (heading_deg >= -180 && heading_deg <=-160)) {
         //turn right
         guidance_h_set_body_vel(cornering_xvel, cornering_yvel);
         guidance_h_set_heading_rate(cornering_turn_rate);
         navigation_state = SAFE;
       } else if  ((heading_deg <= 90 && heading_deg >= 0) || (heading_deg >= -20 && heading_deg <= 0))  {
         //turn left
         guidance_h_set_body_vel(cornering_xvel, -cornering_yvel);
         guidance_h_set_heading_rate(-cornering_turn_rate);
         navigation_state = SAFE;
       } else {
         guidance_h_set_body_vel(cornering_xvel, 0);
         guidance_h_set_heading_rate(RadOfDeg(0));
         navigation_state = SAFE;
       }
       break;
 
 
     case BOTTOM_LINE:
       VERBOSE_PRINT("STATE: BOTTOM_LINE\n");
       // in case we are close to the edge
       // first look at if there are obstacles and we are close to the edge in which case just stop and search for safe heading
       if(cnn_p_center > 0.8) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;  
       } else if((of_b > 0.4 && of_c > 0.4) || (of_b > 0.75f) || (of_b > 0.75f)) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;
       }
 
       // first we check if we have drifted further to the edge of the cyberzoo
       // if there are no obstacles nearby, we look at our current heading and decide to turn either left or right
       if(newy <= -OUTER_BOUNDS && ((90 <= heading_deg  && heading_deg <= 180) || (-180 <= heading_deg && heading_deg <= -90))) {
           navigation_state = OUT_OF_BOUNDS;
       } else if(heading_deg >= -180 && heading_deg <=-70) {
         //turn right
         guidance_h_set_body_vel(cornering_xvel, cornering_yvel);
         guidance_h_set_heading_rate(cornering_turn_rate);
         navigation_state = SAFE;
       } else if (heading_deg <= 180 && heading_deg >= 70) {
         //turn left
         guidance_h_set_body_vel(cornering_xvel, -cornering_yvel);
         guidance_h_set_heading_rate(-cornering_turn_rate);
         navigation_state = SAFE;
       } else {
         guidance_h_set_body_vel(cornering_xvel, 0);
         guidance_h_set_heading_rate(RadOfDeg(0));
         navigation_state = SAFE;
       }
       break;
 
 
     case LEFT_LINE:
       VERBOSE_PRINT("STATE: LEFT_LINE\n");
       // in case we are close to the edge
       // first look at if there are obstacles and we are close to the edge in which case just stop and search for safe heading
       if(cnn_p_center > 0.8) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;  
       } else if((of_b > 0.4 && of_c > 0.4) || (of_b > 0.75f) || (of_b > 0.75f)) {
         guidance_h_set_body_vel(0.f, 0.f);
         navigation_state = SEARCH_HEADING;
         break;
       }
 
       // first we check if we have drifted further to the edge of the cyberzoo
       // if there are no obstacles nearby, we look at our current heading and decide to turn either left or right
       if(newx <= -OUTER_BOUNDS && -180 <= heading_deg && heading_deg <= 0) {
           navigation_state = OUT_OF_BOUNDS;
       } else if((heading_deg >= -90 && heading_deg <=0)  || (heading_deg >= 0 && heading_deg <= 20)){
         //turn right
         guidance_h_set_body_vel(cornering_xvel, cornering_yvel);
         guidance_h_set_heading_rate(cornering_turn_rate);
         navigation_state = SAFE;
       } else if ((heading_deg <= -90 && heading_deg >= -180) || (heading_deg >= 160 && heading_deg <=180)) {
         //turn left
         guidance_h_set_body_vel(cornering_xvel, -cornering_yvel);
         guidance_h_set_heading_rate(-cornering_turn_rate);
         navigation_state = SAFE;
       } else {
         guidance_h_set_body_vel(cornering_xvel, 0);
         guidance_h_set_heading_rate(RadOfDeg(0));
         navigation_state = SAFE;
       }
       break;
 
 
     default:
       break;
   }
   return;
 }
 
 // defining a function which is used when searching for heading based on the history of the obstacles on the left or on the right
 int sign(int num) {
   if(num > 0) {
     return 1;
   } 
   return -1;
 }
 
 
 

 