learning resource: https://pypi.org/project/pyboy/

pyboy api documentation https://docs.pyboy.dk/index.html 
    - so it looks like pyboy is a loadable object. We should be able to initialize controll and read from another script 
    - so this is one of the kwards gameshark (str): GameShark codes to apply. and I think i read somewhere that gameshark is a hack thing which may be helpful for  reading the sf2 values from memory
    - var memory
        Provides a PyBoyMemoryView object for reading and writing the memory space of the Game Boy.

        For a more comprehensive description, see the PyBoyMemoryView class.

        Example:

        >>> pyboy.memory[0x0000:0x0010] # Read 16 bytes from ROM bank 0
        [49, 254, 255, 33, 0, 128, 175, 34, 124, 254, 160, 32, 249, 6, 48, 33]
        >>> pyboy.memory[1, 0x2000] = 12 # Override address 0x2000 from ROM bank 1 with the value 12
        >>> pyboy.memory[0xC000] = 1 # W
        - this will be important for writing memory or deconstructing it


            Provides a MemoryScanner object for locating addresses of interest in the memory space of the Game Boy. This might require some trial and error. Values can be represented in memory in surprising ways.

            Open an issue on GitHub if you need finer control, and we will take a look at it.

            Example:

            >>> current_score = 4 # You write current score in game
            >>> pyboy.memory_scanner.scan_memory(current_score, start_addr=0xC000, end_addr=0xDFFF)
            []
            >>> for _ in range(175):
            ...     pyboy.tick(1, True) # Progress the game to change score
            True...
            >>> current_score = 8 # You write the new score in game
            >>> from pyboy.api.memory_scanner import DynamicComparisonType
            >>> addresses = pyboy.memory_scanner.rescan_memory(current_score, DynamicComparisonType.MATCH)
            >>> print(addresses) # If repeated enough, only one address will 
    
    - this seems pretty important in combo with the button arguments that will let us pause and take action after the neural net processes 
        - def save_state
            (
            self, file_like_object)
            Saves the complete state of the emulator. It can be called at any time, and enable you to revert any progress in a game.

            You can either save it to a file, or in-memory. The following two examples will provide the file handle in each case. Remember to seek the in-memory buffer to the beginning before calling PyBoy.load_state():

            >>> # Save to file
            >>> with open("state_file.state", "wb") as f:
            ...     pyboy.save_state(f)
            >>>
            >>> # Save to memory
            >>> import io
            >>> with io.BytesIO() as f:
            ...     f.seek(0)
            ...     pyboy.save_state(f)
            0
            
    - im not sure how but this will be important def hook_register
    (
    self, bank, addr, callback, context)

    - this will be super important for getting the gamestate values https://gbdev.io/pandocs/Memory_Map.html 

    - 