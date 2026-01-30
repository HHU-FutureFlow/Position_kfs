from struck.strukt import xywh
def filtter(last, current, alpha = 0.0):
    return alpha*last + (1-alpha)*current

def initialize(last_results,current_results):
    if last_results == None:
        return current_results

    final_results = xywh()
    final_results.w = filtter(last_results.w,current_results.w, 0.3)
    final_results.h = filtter(last_results.h, current_results.h, 0.3)
    final_results.x = filtter(last_results.x, current_results.x, 0.3)
    final_results.y = filtter(last_results.y, current_results.y, 0.3)

    return final_results