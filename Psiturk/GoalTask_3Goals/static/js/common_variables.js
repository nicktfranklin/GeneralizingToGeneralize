/**
 * Created by nickfranklin on 2/2/16.
 */


// Start by defining the action maps. These consists of the right hand keys (j,k,l,;) and the
// left hand keys (a,s,d,f) as well as an order of cardinal directions ('right', 'down', 'left', 'up')
// and ('up', 'left', 'down', 'right'). For counter balance, we want to randomly assign each key set
// (ii.e. right hand vs left hand) one of the cardinal direction orders. Then we will call one of these
// mappings "action_mapping_a" for the high frequency mappig and "action_mappig_b" for the low frequency
// maps



/* firefox handles javascript keyboard codes differently than safari/chrome.
check which browser and set the keys accordingly*/
var left_hand_keys = [65, 83, 68, 70];
var right_hand_keys;
if (navigator.userAgent.indexOf('Firefox') != -1) {
    right_hand_keys = [74, 75, 76, 59];
} else {
    right_hand_keys = [74, 75, 76, 186]
}


var cardinal_order_a = ['left','up','down','right'];
var cardinal_order_b = ['up','left','right','down'];

var cardinal_orders = _.shuffle([cardinal_order_a, cardinal_order_b]);

// create a left hand and right hand set (that are labeled) for the test trials
var action_mapping_lh = {};
var action_mapping_rh = {};

for (var ii=0; ii<4; ii++) {
    action_mapping_lh[left_hand_keys[ii]] = cardinal_orders[1][ii];
    action_mapping_rh[right_hand_keys[ii]] = cardinal_orders[0][ii];

}

// these instruction sets math the hands and are used in the tutorial and in the test phase
var instruction_set_lh = '<b>a</b>, <b>s</b>, <b>d</b>, and <b>f</b>';
var instruction_set_rh = '<b>j</b>, <b>k</b>, <b>l</b>, and <b>;</b>';
var instruction_set_lh_red = '<b><span style="color: #FF0000">a</span></b>, ' +
        '<b><span style="color: #FF0000">s</span></b>, <b><span style="color: #FF0000">d</span></b>, and' +
        ' <b><span style="color: #FF0000">f</span></b>';
var instruction_set_rh_red = '<b><span style="color: #FF0000">j</span></b>, ' +
        '<b><span style="color: #FF0000">k</span></b>, <b><span style="color: #FF0000">l</span></b>, and' +
        ' <b><span style="color: #FF0000">;</span></b>';

// flip a coin to assign the mapping to an "a" mapping and a "b" mapping . This randomizes the context assocation with
// either right hand or left hand.
var action_mapping_a;
var action_mapping_b;
var instruction_set_a;
var instruction_set_b;
var instruction_set_b_red;

if (Math.random() > .5) {
	action_mapping_a = action_mapping_lh;
	action_mapping_b = action_mapping_rh;
    instruction_set_a = instruction_set_lh;
    instruction_set_b = instruction_set_rh;
    instruction_set_b_red = instruction_set_rh_red;
} else {
	action_mapping_a = action_mapping_rh;
	action_mapping_b = action_mapping_lh;
    instruction_set_a = instruction_set_rh;
    instruction_set_b = instruction_set_lh;
    instruction_set_b_red = instruction_set_lh_red;
}



// /// define colors
// var list_colors = [
// 	//'#F0F8FF', // alice blue (close to white!)
// 	//'#FAEBD7', // antique white
// 	'#00FFFF', // aqua
// 	'#7FFFD4', // Aquamarine
// 	'#FFE4C4', // Bisque
// 	'#0000FF', // Blue
// 	'#8A2BE3', // Blue viloet
// 	'#5F9EA0', // cadet blue
// 	'#7FFF00', // Chartreuse
// 	'#D2691E', // chocolate
// 	'#FF7F50', // coral (red-pink)
// 	'#DC143C', // crimson
// 	'#0000BB', // dark blue
// 	'#B8860B', // dark goldenrod
// 	'#A9A9A9', // dark grey
// 	'#006400', // dark green
// 	'#FF8C00', // dark orange
// 	'#9932CC', // dark orchid
// 	'#8B0000', // dark red
// 	'#FF1493', // deep pink
// 	'#696969', // dim grey
// 	'#1E90FF', // dodger blue
// 	'#B22222', // fire brick
// 	'#229B22', // forest green
// 	'#FF00FF', // Fuchsia
// 	'#FFD700', // gold
// 	'#808000', // olive
// 	'#FF4500', // orange red
// 	'#2E8B58', // sea green
// 	'#A0522D' // sienna
// ];
var list_colors = [
    '#FFC0CB', // Pink
    '#FF1493', // Deep Pink
    '#DDA0DD', // Lavender
    '#FF00FF', // Fuschia
    '#9400D3', // Dark Violet
    '#F08080', // Choral
    '#DC143C', // crimson
    '#8B0000', // dark red
    '#FF8C00', // dark orange
    '#FF4500', // orange red
    '#FFD700', // gold
    '#7FFF00', // Chartreuse
    '#229B22', // forest green
    '#00FFFF', // cyan
    '#00BFFF', // DeepSky Blue
    '#0000CD', // Medium Blue
    '#8B4513', // Saddle Brown
    '#696969' // Dim Grey
];

list_colors = _.shuffle(list_colors);

var tutorial_color_1 = list_colors.shift();
var tutorial_color_2 = list_colors.shift();

var wall_probability = 0.75; // there is some random probability of seeing walls in the first place.
var wall_probability_generalization = 0.75;
