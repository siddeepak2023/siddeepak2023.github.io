import java.util.Scanner;

/**
 * Guessing game - In class version
 * 
 * Write the methods that make this game work
 * Use try and catch for potential errors
 * You can use a do...while loop to make sure a while loop runs at least once
 * Null Pointer Exceptions? Variables initialize to null until a value is set
 */
public class GuessingGame {
    // Program Variables
    static Scanner scanner = new Scanner(System.in);
    static final int MIN_GUESS = 1;
    static final int MAX_GUESS = 10;

    // Game Variables
    static int answer;
    static int[] guessedNumbers;

    public static void main(String[] args) {
        boolean game = playGame(), running = game;

        while(game) {
            init();
            while(running) {
                if(getGuess() == answer) {
                    System.out.println("You WIN!!!");
                    running = false;
                } else if(getCompGuess() == answer) {
                    System.out.println("Computer WINS!!");
                    running = false;
                } else {
                    System.out.println("No one guessed correctly...");
                }
            }
            game = playGame();
            running = game;
        }
        scanner.close();
    }
    
    /**
     * Initializes the game states
     * Populates game variables
     */
    private static void init() {
        //TODO: write this method
    }

    /**
     * Asks the player if they want to play a game
     * @return boolean value of the answer
     */
    private static boolean playGame() {
        //TODO: write this method
        return false;
    }

    /**
     * Asks the player for a guess
     * Must not allow the player to crash the game
     * @return int of the player's guess
     */
    private static int getGuess() {
        //TODO write this method
        return -1;
    }

    /**
     * Gets a guess for the computer
     * Must not select a value that has already been guessed be either the
     * player or the computer
     * @return int of the computer's guess
     */
    private static int getCompGuess() {
        //TODO write this method
        return -1;
    }
}