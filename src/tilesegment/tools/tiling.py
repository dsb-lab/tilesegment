def compute_closest_larger_multiple(number, multiple):
    if not (number < 0 or multiple <= 0):
        closest_multiple = (number // multiple) * multiple  # Find the largest multiple less than or equal to the number
        if closest_multiple < number:
            closest_multiple += multiple
        return closest_multiple
    else: 
        raise Exception("number and multiple cannot be negative. Multiple cannot be 0")

def pad_image_and_square_array(IMG, final_size):
    return None
