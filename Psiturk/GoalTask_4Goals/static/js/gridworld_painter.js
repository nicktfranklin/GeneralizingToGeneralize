"use strict";

var GridWorldPainter = function (context) {
    this.TILE_SIZE = context.gridworld.tile_size;
	this.COLORS = context.colors;
	this.AGENT_WIDTH = this.TILE_SIZE*.3;
	this.WALL_STROKE_SIZE = 10;
	this.ACTION_ANIMATION_TIME = 150;
	this.CONTRACT_SIZE = .9;
	this.SUBACTION_PROPORTION = .2;
	this.SUBACTION_DISTANCE = .2;
	this.SUBACTION_COLLISION_PROPORTION = .7;
	this.SUBACTION_COLLISION_DISTANCE = .7;
	this.REWARD_DISPLAY_TIME = 800;

	this.gridworld = context.gridworld;
	this.animationTimeouts = [];
	this.objectImages = {};
	this.currentAnimations = {};
    this.draw_goals = context.draw_goals;
};


GridWorldPainter.prototype.init = function (container) {
	this.width = this.TILE_SIZE*this.gridworld.width;
	this.height = this.TILE_SIZE*this.gridworld.height;
	this.paper = Raphael(container, this.width, this.height);

	//assign colors to agents
	this.AGENT_COLORS = {};
	for (var i = 0; i < this.gridworld.agents.length; i++){
		this.AGENT_COLORS[this.gridworld.agents[i].name] = this.COLORS[i];
	}

	this.walls = {};
	this.goals = {};

    //draw tiles
	for (var x = 0; x < this.gridworld.width; x++) {
		for (var y = 0; y < this.gridworld.height; y++) {
			var tile_params = {
				type : 'rect',
				x : x*this.TILE_SIZE,
				y : this.height - (y+1)*this.TILE_SIZE,
				width : this.TILE_SIZE,
				height : this.TILE_SIZE,
				stroke : 'black',
				fill: '#FFFFFF'
			};
			//goals
            if (this.draw_goals) {
                for (var g = 0; g < this.gridworld.goals.length; g++) {
                    var goal = this.gridworld.goals[g];
                    this.goals[goal.location] = goal; //store locations of all goals for animations
                    if (String(goal.location) === String([x,y])) {
                        tile_params.fill = this.AGENT_COLORS[goal.agent];
                    }
                }
            }

			//walls
			for (var w = 0; w < this.gridworld.walls.length; w++) {
				var wall = this.gridworld.walls[w];
				if (String([wall[0], wall[1]]) == String([x,y])) {
					this.walls[wall] = 1; //store locations of all walls for animations
					if (wall.length == 2) {
						tile_params.fill = 'black';
					}
					else if (wall.length == 3) {
						var from, to;
						switch (wall[2]) {
							case 'up':
								from = [0, 1];
								to = [1, 1];
								break;
							case 'down':
								from = [0,0];
								to = [1, 0];
								break;
							case 'left':
								from = [0,0];
								to = [0, 1];
								break;
							case 'right':
								from = [1, 0];
								to = [1, 1];
								break
						}
						from = [(x+from[0])*this.TILE_SIZE, (this.gridworld.height - (from[1] + y))*this.TILE_SIZE];
						to = [(x+to[0])*this.TILE_SIZE, (this.gridworld.height - (to[1] + y))*this.TILE_SIZE];

						var wall_path = 'M'+from.join(' ') + 'L'+to.join(' ');
						var wall_img = this.paper.path(wall_path);
						wall_img.attr({"stroke-width" : this.WALL_STROKE_SIZE, stroke : 'black'});
					}
				}
			}
			//var rect = new paper.Path.Rectangle(params);

			var tile = this.paper.add([tile_params])[0];
		}
	}

    // draw goal labels
    if (this.draw_goals) {
        for (g = 0; g < this.gridworld.goals.length; g++) {
            goal = this.gridworld.goals[g];
            var params = {
                type: 'text',
                x: (goal.location[0] + .5) * this.TILE_SIZE,
                y: (this.gridworld.height - goal.location[1] - .5) * this.TILE_SIZE,
                text: goal.display_label,
                "font-size": 36,
                stroke : 'black',
                fill: '#FFFFFF'};
            var r = this.paper.add([params])[0];
        }
    }
};

GridWorldPainter.prototype.drawState = function (state) {
	// console.log(this);
	//paper.getPaper(this.canvas);
	for (var object in state) {
		if (!state.hasOwnProperty(object)) {
			continue
		}
		object = state[object];

		//draw agents
		if (object.type == 'agent') {
			var agent_params = {cx : (object.location[0] + .5)*this.TILE_SIZE,
				                cy : (this.gridworld.height - object.location[1] - .5)*this.TILE_SIZE};
			if (this.objectImages.hasOwnProperty(object.name)) {
				this.objectImages[object.name].attr(agent_params);
			}
			else {
				agent_params.type = 'circle';
				agent_params.fill = this.AGENT_COLORS[object.name];
				agent_params.r = this.AGENT_WIDTH;
				agent_params['stroke'] = 'white';
				agent_params['stroke-width'] = 1;

				this.objectImages[object.name] = this.paper.add([agent_params])[0];
			}
		}
	}
};

//cases:
//normal move
//wait
//hit wall
//hit edge of world
//hit another agent (1) who waits; (2) who is also moving to the same tile

GridWorldPainter.prototype.drawTransition = function (state, action, nextState, mdp) {
	console.log('------------');
	this.drawState(state);
    var agent;

	var intendedLocation = {};
	for (agent in state) {
	    if (state.hasOwnProperty(agent)) {
            if (state[agent].type !== 'agent') {
                continue
            }
            if (typeof mdp == 'undefined') {
                intendedLocation[agent] = nextState[agent];
            }
            else {
                intendedLocation[agent] = mdp.getNextIntendedLocation(state[agent].location, action[agent]);
            }
        }
	}

	for (agent in state) {
		if (!state.hasOwnProperty(agent) || state[agent].type !== 'agent') {
			continue
		}
		//waiting
		if (action[agent] == 'wait') {
			console.log(agent + ' waits');
			var expand = (function (painter, agentImage, original_size, time) {
				return function () {
					var anim = Raphael.animation({r : original_size}, time,	'backOut');
					agentImage.animate(anim);
					$.subscribe('killtimers', painter.makeTimerKiller(agentImage, anim))
				}
			})(this, this.objectImages[agent], this.objectImages[agent].attr('r'), this.ACTION_ANIMATION_TIME*.5);

			var contract = Raphael.animation({r : this.objectImages[agent].attr('r')*this.CONTRACT_SIZE},
                this.ACTION_ANIMATION_TIME*.5, 'backIn', expand);
			this.objectImages[agent].animate(contract);

			//attach to killtimer
			$.subscribe('killtimers', this.makeTimerKiller(this.objectImages[agent], contract))
		}
		//try and fail
		else if (String(intendedLocation[agent]) !== String(nextState[agent].location)) {
			console.log(agent +' tried and failed');

			//distance to try depends on failure condition
			//1 - hit a wall or just hit another agent waiting
			var SUBACTION_DISTANCE = this.SUBACTION_DISTANCE;
			var SUBACTION_PROPORTION = this.SUBACTION_PROPORTION;
			for (var otherAgent in state){
				if (!state.hasOwnProperty(otherAgent) || state[otherAgent].type !== 'agent' || agent == otherAgent) {
					continue
				}

				//2 - 2 agents try to get into the same spot
				if ((String(intendedLocation[agent]) == String(nextState[otherAgent].location) ||
					String(intendedLocation[agent]) == String(intendedLocation[otherAgent]))
					&&
					!(String(intendedLocation[agent]) == String(nextState[otherAgent].location) &&
					String(intendedLocation[otherAgent]) == String(nextState[agent].location))) {
					console.log(agent +' collided with ' + otherAgent);
					SUBACTION_DISTANCE = this.SUBACTION_COLLISION_DISTANCE;
					SUBACTION_PROPORTION = this.SUBACTION_COLLISION_PROPORTION;
				}
			}

			var moveBack = (function (painter, agentImage, original_x, original_y, time) {
				return function () {
					var anim = Raphael.animation({cx : original_x, cy: original_y}, time, 'backOut');
					agentImage.animate(anim);
					$.subscribe('killtimers', painter.makeTimerKiller(agentImage, anim))
				}
			})(this, this.objectImages[agent], this.objectImages[agent].attr('cx'), this.objectImages[agent].attr('cy'),
				this.ACTION_ANIMATION_TIME*SUBACTION_PROPORTION);

			var new_x = (intendedLocation[agent][0]*SUBACTION_DISTANCE) +
						(state[agent].location[0]*(1 - SUBACTION_DISTANCE));

			var new_y = ((this.gridworld.height - intendedLocation[agent][1])*SUBACTION_DISTANCE)
						+ (this.gridworld.height - state[agent].location[1])*(1 - SUBACTION_DISTANCE);

			new_x = (new_x + .5)*this.TILE_SIZE;
			new_y = (new_y - .5)*this.TILE_SIZE;

			var moveToward = Raphael.animation({cx : new_x, cy : new_y},
												this.ACTION_ANIMATION_TIME*(1-SUBACTION_PROPORTION), 'backIn',
												moveBack);

			this.objectImages[agent].animate(moveToward);
			$.subscribe('killtimers', this.makeTimerKiller(this.objectImages[agent], moveToward));
		}
		//normal movement
		else {
			console.log(agent + ' makes normal movement');
			var movement = Raphael.animation(
                {
                    cx : (nextState[agent].location[0] + .5)*this.TILE_SIZE,
                    cy : (this.gridworld.height - nextState[agent].location[1] - .5)*this.TILE_SIZE
                },
                this.ACTION_ANIMATION_TIME, 'easeInOut');
            this.objectImages[agent].animate(movement);
			$.subscribe('killtimers', this.makeTimerKiller(this.objectImages[agent], movement))
		}
	}


};

GridWorldPainter.prototype.showReward = function (loc, agent, text) {
	var params = {type : 'text',
							 x : (loc[0] + .5)*this.TILE_SIZE,
							 y : (this.gridworld.height - loc[1] - .5)*this.TILE_SIZE,
							 text : text,
							 "font-size" : 30,
                             "font-weight": "bold",
							 stroke: 'black',
							 fill : 'yellow'};
	var r = this.paper.add([params])[0];
	var r_animate = Raphael.animation({y : r.attr("y") - .5*this.TILE_SIZE, opacity : 0}, this.REWARD_DISPLAY_TIME);

	r.animate(r_animate);
	$.subscribe('killtimers', this.makeTimerKiller(r, r_animate))
};

GridWorldPainter.prototype.showLoss = function (loc, agent, text) {
	var params = {type : 'text',
		x : (loc[0] + .5)*this.TILE_SIZE,
		y : (this.gridworld.height - loc[1] - .5)*this.TILE_SIZE,
		text : text,
		"font-size" : 30,
		// "font-weight": "bold",
		stroke: 'black',
		fill : 'black'};
	var r = this.paper.add([params])[0];
	var r_animate = Raphael.animation({y : r.attr("y") - .5*this.TILE_SIZE, opacity : 0}, this.REWARD_DISPLAY_TIME);

	r.animate(r_animate);
	$.subscribe('killtimers', this.makeTimerKiller(r, r_animate))
};


//GridWorldPainter.prototype.drawGoalLabel = function (loc, agent, text) {
//	var params = {type : 'text',
//							 x : (loc[0] + .5)*this.TILE_SIZE,
//							 y : (this.gridworld.height - loc[1] - .5)*this.TILE_SIZE,
//							 text : text,
//							 "font-size" : 36,
//							 stroke : this.AGENT_COLORS[agent],
//							 fill : 'yellow'};
//	var r = this.paper.add([params])[0];
//	//var r_animate = Raphael.animation({y : r.attr("y") - .5*this.TILE_SIZE, opacity : 0}, this.REWARD_DISPLAY_TIME)
//
//	//r.animate(r_animate);
//	//$.subscribe('killtimers', this.makeTimerKiller(r, r_animate))
//};


GridWorldPainter.prototype.makeTimerKiller = function (element, animation) {
		return function () {
			element.stop(animation);
		}
	};

GridWorldPainter.prototype.remove = function () {
	this.paper.remove();
};
