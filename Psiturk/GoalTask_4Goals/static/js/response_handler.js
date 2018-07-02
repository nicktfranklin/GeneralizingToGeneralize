/**
 * Created by nick on 7/6/15.
 *
 * This is intended to be a separate piece of code that allows me to swap in
 * different keyboard response mappings.
 */
"use strict";

var wait_before_actions_time;
var allow_response;
wait_before_actions_time = 0;

var demo_responseHandlerGenerator;
demo_responseHandlerGenerator = function (action_mapping) {

    return function (event) {
        // Use the Action map to translate the action correctly.
        var response;
        response = action_mapping[event.which];
        if (response === undefined) {
            response = 'wait';
        }

        var rt = new Date().getTime() - trial_on; // record the reaction time.

        //////
        var agentActions = {};

        agentActions['agent1'] = response;

        var nextState = this.mdp.getTransition(this.state, agentActions);
        this.painter.drawTransition(this.state, agentActions, nextState, this.mdp);
        this.state = nextState;

        $(document).unbind('keydown.gridworld');
        this.restart = false;

        /* Determine the Rewards */
        //var reset_key_time = this.painter.ACTION_ANIMATION_TIME;
        for (var agent in this.state) {
            if (this.state.hasOwnProperty(agent)) {
                if (this.mdp.inGoal(nextState[agent]['location'], agent)) {

                    var goal_value = this.mdp.getStateValue(nextState[agent]['location'], agent);
                    var display;
                    move_to_next_trial = true;


                    // var reward_label;
                    if (goal_value > 0) {
                        // reward_label = 'Great!';
                        display = 'You got to the right goal!<br><br> ' +
                            '<I><span style="color #707070">Press enter to continue</span></I>';

                    } else {
                        this.restart = true;
                        // reward_label = 'Try Again!';
                        display = "Oops! That's not the right goal!<br><br> " +
                            '<I><span style="color #707070">Press enter to try again</span></I>';
                    }

                    // if agent is in goal state, celebrate.
                    var celebrateGoal = (function (painter, location, agent) {
                        return function () {
                            if (goal_value) {
                                painter.showReward(location, agent, 'Great!');
                            } else {
                                painter.showLoss(location, agent, 'Try Again!')
                            };

                            if (typeof painter.points === 'undefined') {
                                painter.points = {'agent1': 0}
                            }
                            painter.points[agent]++;
                            $('#trial_text').html(display);
                        }
                    })(this.painter, nextState[agent]['location'], agent);

                    var th;
                    th = setTimeout(celebrateGoal, this.painter.ACTION_ANIMATION_TIME);
                    $.subscribe('killtimers', (function (th) {
                            return function () {
                                clearTimeout(th)
                            }
                        })(th)
                    );

                    console.log("check end");
                }
            }
        }

        // // Record in psiturk
        // psiTurk.recordTrialData(
        //     {
        //         'phase': 'Demo Trial',
        //         'key-press': event.which,
        //         'action': response,
        //         'End Location': this.state['agent1'].location,
        //         'rt': rt,
        //         'action_map': action_mapping,
        //         'In Goal': this.mdp.inGoal(nextState[agent]['location'], agent)
        //     });

        var reset_key_handler;
        if (this.mdp.inGoal(nextState[agent]['location'], agent)) {
            reset_key_handler = (function () {
                return function () {
                    $(document).bind('keydown.gridworld', function (event) {});
                }
            })();
        } else {
            //note: you need a closure in order to properly reset
            reset_key_handler = (function (key_handler) {
                return function () {
                    $(document).bind('keydown.gridworld', key_handler);
                }
            })(this.key_handler);
        }


        //var th;
        th = setTimeout(reset_key_handler, this.painter.ACTION_ANIMATION_TIME +
          wait_before_actions_time);
        $.subscribe('killtimers', (function (th) {
                return function () {
                    clearTimeout(th)
                }
            })(th)
        );

        trial_on = new Date().getTime();
    };

};


var demo_responseHandler_generator_noReachableAction = function(action_mapping) {

    // allow_response = true;
    var n_responses = 0;

    return function (event) {
        // count the number of responses to determine if enough moves have been made to continue
        n_responses++;

        // allow the subject to exit the trial after seven responses (if subject hits enter, skip rest of code)
        if (n_responses >= 7 && event.which == 13) {
            return;
        }

        // allow the subject to exit the trial after seven responses
        if (n_responses >= 7) {
            $('#trial_text').html('Great!<br> <br> <I><span style="color: #707070">Press enter to continue</span></br>');
            move_to_next_trial = true;
        }



        // Use the Action map to translate the action correctly.
        var response;
        response = action_mapping[event.which];
        if (response === undefined) {
            response = 'wait';
        }

        var rt = new Date().getTime() - trial_on; // record the reaction time.

        ////choose random actions for other agents
        var agentActions = {};
        var availableActions = ['left', 'up', 'right', 'down', 'wait'];
        for (var agent in this.state) {
            if (this.state[agent].type == 'agent' && agent !== 'agent1') {
                agentActions[agent] = availableActions[Math.floor(Math.random() * availableActions.length)]
            }
        }
        agentActions['agent1'] = response;

        var nextState = this.mdp.getTransition(this.state, agentActions);
        this.painter.drawTransition(this.state, agentActions, nextState, this.mdp);
        this.state = nextState;

        $(document).unbind('keydown.gridworld');

        // // Record in psiturk (work in progress)
        // psiTurk.recordTrialData(
        //     {
        //         'phase': 'Demo Trial',
        //         'key-press': event.which,
        //         'action': response,
        //         'End Location': this.state['agent1'].location,
        //         'rt': rt, //
        //         'action_map': '', // action_mapping,
        //         'In Goal': this.mdp.inGoal(nextState[agent]['location'], agent)
        //     }
        // );

        //note: you need a closure in order to properly reset
        var reset_key_handler = (function (key_handler) {
            return function () {
                $(document).bind('keydown.gridworld', key_handler);
        }
        })(this.key_handler);



        var th = setTimeout(reset_key_handler, this.painter.ACTION_ANIMATION_TIME + wait_before_actions_time);
        $.subscribe('killtimers', (function (th) {
                return function () {
                    clearTimeout(th)
                }
            })(th)
        );


        trial_on = new Date().getTime();


    };

};


var demo_responseHandler_generator_endDemo= function(action_mapping) {

    var n_responses = 0;

    return function (event) {

        n_responses++;
        move_to_next_trial = true;

        // allow the subject to exit the trial after seven responses (if subject hits enter, skip rest of code)
        if (event.which == 13) {
            return;
        }

        // allow the subject to exit the trial after seven responses
        if (n_responses >= 4) {
            $('#trial_text').html("That's how it works!!!<br>.~*`*~.~*`*~.~*`*~.~*`*~.<br> <I><span style='color: #707070'>Press enter to continue</span></br>");
        }


        // Use the Action map to translate the action correctly.
        var response;
        response = action_mapping[event.which];
        if (response === undefined) {
            response = 'wait';
        }

        var rt = new Date().getTime() - trial_on; // record the reaction time.

        ////choose random actions for other agents
        var agentActions = {};
        var availableActions = ['left', 'up', 'right', 'down', 'wait'];
        for (var agent in this.state) {
            if (this.state[agent].type == 'agent' && agent !== 'agent1') {
                agentActions[agent] = availableActions[Math.floor(Math.random() * availableActions.length)]
            }
        }
        agentActions['agent1'] = response;

        var nextState = this.mdp.getTransition(this.state, agentActions);
        this.painter.drawTransition(this.state, agentActions, nextState, this.mdp);
        this.state = nextState;

        $(document).unbind('keydown.gridworld');

        // Record in psiturk (work in progress)
        psiTurk.recordTrialData(
            {
                'phase': 'Demo Trial',
                'key-press': event.which,
                'action': response,
                'End Location': this.state['agent1'].location,
                'rt': rt,
                'action_map': action_mapping,
                'In Goal': this.mdp.inGoal(nextState[agent]['location'], agent)
            }
        );

        //note: you need a closure in order to properly reset
        var reset_key_handler = (function (key_handler) {
            return function () {
                $(document).bind('keydown.gridworld', key_handler);
            }
        })(this.key_handler);



        var th = setTimeout(reset_key_handler, this.painter.ACTION_ANIMATION_TIME + wait_before_actions_time);
        $.subscribe('killtimers', (function (th) {
                return function () {
                    clearTimeout(th)
                }
            })(th)
        );


        trial_on = new Date().getTime();


    };

};


var responseHandlerGenerator;
responseHandlerGenerator = function (action_mapping) {
    /**
     * Makes the response handler for the experimental trials
     * @param action_mapping
     * @returns {Function}
     */

    return function (event) {

        // Use the Action map to translate the action correctly.
        var response = action_mapping[event.which];
        if (response === undefined) {
            response = 'wait';
        }

        var rt = new Date().getTime() - trial_on; // record the reaction time.

        ////choose random actions for other agents --- don't need this.
        var agentActions = {};
        agentActions['agent1'] = response;

        var nextState = this.mdp.getTransition(this.state, agentActions);
        var startLocation = this.state['agent1'].location;

        this.painter.drawTransition(this.state, agentActions, nextState, this.mdp);
        this.state = nextState;
        this.actions_taken++;
        var goal_value = 0;
        var goal_id = 'None'; // this is the goal's subject-independent label (has the same statistics across subjects)
        var goal_display_label = 'None'; // this is the goal's label on the screen (this is randomized)

        $(document).unbind('keydown.gridworld');

        /* Determine the Rewards */
        //var reset_key_time = this.painter.ACTION_ANIMATION_TIME;
        for (var agent in this.state) {

            if (this.mdp.inGoal(nextState[agent]['location'], agent)) {

                // allow_response = false;
                move_to_next_trial = true;
                // trial_complete = true;

                //get the value, identity and on-screen label of the goal
                goal_value = this.mdp.getStateValue(nextState[agent]['location'], agent);
                goal_display_label = this.mdp.getGoalDisplayLabel(nextState[agent]['location'], agent);
                goal_id = this.mdp.getGoalID(nextState[agent]['location'], agent);
                this.total_points += goal_value;
                console.log("Goal: " + goal_id + ", Label: " + goal_display_label);

                // if agent is in goal state, celebrate.
                var celebrateGoal = (function (painter, location, agent, points,
                   total_points, goal_display_label, context) {
                    return function () {
                        if (goal_value > 0) {
                            painter.showReward(location, agent, '+'.concat(String(points)));
                        } else {
                            painter.showLoss(location, agent, ''.concat(String(points)));

                        }


                        var text_display = 'You chose goal: ' +
                          '<span style="font-weight: bold"><span style="font-size:150%">' +
                            // '<span style="color:' + painter.AGENT_COLORS['agent1'] +'">' +
                            goal_display_label +'</span></span> in ' +
                            '<span style="font-size:150%"><span style="font-weight: bold">' +
                            '<span style="color:' + painter.AGENT_COLORS['agent1'] +'">' +
                            'Room ' + String(context) + '</span></span></span><br> and won: <span style="color:' +
                            painter.AGENT_COLORS['agent1'] +'"><span style="font-size:150%">' +
                            '<span style="font-weight: bold">' + String(goal_value)
                            +  '</span></span></span> ' + 'points (' + String(total_points) + ' total)<br>' +
                            '<span style="font-style: italic"><span style="color: #707070">Press enter to continue' +
                            '</span></span>';

                        $('#trial_text').html(text_display);
                    }
                })(this.painter, nextState[agent]['location'], agent, goal_value,
                  this.total_points, goal_display_label, this.context);

                var th;
                // if (goal_value > 0) {
                th = setTimeout(celebrateGoal, this.painter.ACTION_ANIMATION_TIME);
                // } else {
                    th = setTimeout(function() {}, this.painter.ACTION_ANIMATION_TIME);
                // }
                $.subscribe('killtimers', (function (th) {
                        return function () {
                            clearTimeout(th)
                        }
                    })(th)
                );

            } else {
                var text_display = 'Which goal is the best?<span style="font-size:150%"></span><br>' +
                    '<span style="color:' + this.painter.AGENT_COLORS['agent1'] +'">' +
                    // '<span style="text-shadow: 2px 2px #ff0000">'
                    '<span style="font-size:150%"><span style="font-weight: bold">' +
                    'Room '+ this.context + '</span></span></span>' +
                    // '</span>' +
                    '<br> ' +
                    '<span style="color: #707070"><span style="font-style: italic">Use the <b>a</b>, <b>s</b>,' +
                    ' <b>d</b>, <b>f</b>, and <b>j</b>, <b>k</b>, <b>l</b>, <b>;</b> keys to move.</span></span>';

                $('#trial_text').html(text_display);
            }
        }

        // Record in psiturk (work in progress)
        // run test conditions to reduce size of information in the data:


        var goal_locations_to_save;
        if (this.actions_taken == 1) {
            goal_locations_to_save = this.mdp.getGoalLocations(agent);
        } else {

            goal_locations_to_save = [];
        }

        var walls_to_save;
        var mapping_to_save;
        var color_to_save;
        if (this.times_seen_context == 1) {
          mapping_to_save = action_mapping;
          walls_to_save = this.gridworld.walls;
          color_to_save = this.painter.AGENT_COLORS['agent1'];
        } else {
          walls_to_save = [];
          mapping_to_save = [];
          color_to_save = [];
        }

        psiTurk.recordTrialData(
            {
                'Context': this.context,
                'Start Location': startLocation,
                'Key-press': event.which,
                'End Location': this.state['agent1'].location,
                'Action Map': mapping_to_save,
                'Walls': walls_to_save,
                'Action': response, // this is the cardinal direction taken
                'Reward': goal_value,
                'In Goal': this.mdp.inGoal(nextState[agent]['location'], agent),
                'Chosen Goal': goal_id,
                'Displayed Goal Label': goal_display_label,
                'Steps Taken': this.actions_taken,
                'Goal Locations': goal_locations_to_save,
                'Trial Number': trial_number,
                'Times Seen Context': this.times_seen_context,
                'phase': 'Experiment',
                'rt': rt,
                // 'n actions taken': this.actions_taken,
                // these are general trial information
                'agent_color': color_to_save,
            });


        //note: you need a closure in order to properly reset
        // goal check, then reset key handler
        var reset_key_handler;
        if (this.mdp.inGoal(nextState[agent]['location'], agent)) {
            reset_key_handler = (function () {
                return function () {
                    $(document).bind('keydown.gridworld', function (event) {});
                }
            })();
        } else {
            reset_key_handler = (function (key_handler) {
                return function () {
                    $(document).bind('keydown.gridworld', key_handler);
                }
            })(this.key_handler);
        }


        //var th;
        th = setTimeout(reset_key_handler, this.painter.ACTION_ANIMATION_TIME + wait_before_actions_time);
        $.subscribe('killtimers', (function (th) {
                return function () {
                    clearTimeout(th)
                }
            })(th)
        );
        trial_on = new Date().getTime();
    };
};
