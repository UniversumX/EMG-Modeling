// 1 is for move up, -1 is for move down, nothing for hold/stationary in this situation.
int move_binary(float bicep_signal, float tricep_signal) {
    bool bicep_on = (bicep_signal > 500) && (bicep_signal < 3500);
    bool tricep_on = (tricep_signal > 500) && (tricep_signal < 3500);
    if (bicep_on || tricep_on) return 1;
    return -1;
}

// 1 is for move up, -1 is for move down, 0 is for stay where you are
int move_ternary(float bicep_signal, float tricep_signal) {
    bool bicep_on = (bicep_signal > 500) && (bicep_signal < 3500);
    bool tricep_on = (tricep_signal > 500) && (tricep_signal < 3500);
    if (bicep_on && tricep_on) return bicep_signal >= tricep_signal ? 1 : -1;
    if (bicep_on) return 1;
    if (tricep_on) return -1;
    return 0;
}
