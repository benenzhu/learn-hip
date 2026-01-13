def log(*msg) -> None:
    import time
    import sys
    import os
    try:
        raise Exception
    except:
        linenum = sys.exc_info()[2].tb_frame.f_back.f_lineno
        filename = sys.exc_info()[2].tb_frame.f_back.f_code.co_filename

    # ANSI color codes
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    filename_only = filename.split("/")[-1]
    current_time = time.strftime("%H:%M:%S", time.localtime())
    milliseconds = int((time.time() % 1) * 1000)
    time_with_ms = f"{current_time}.{milliseconds:03d}"
    print(f'{time_with_ms} {BLUE}{os.getcwd()}/{filename_only}{RESET}:{YELLOW}{linenum}{RESET}:', *msg)
    print('', end = "", flush=True)