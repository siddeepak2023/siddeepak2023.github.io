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
    static final int MAX_GUESS = 50;

    // Game Variables
    static int answer;
    static int[] guessedNumbers = new int[MAX_GUESS+2];

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
        answer = (int) (Math.random()*(MAX_GUESS-MIN_GUESS)+MIN_GUESS);

    }

    /**
     * Asks the player if they want to play a game
     * @return boolean value of the answer
     */
    private static boolean playGame() {
        System.out.println("press y to play game");
        String userinput = scanner.nextLine();
        if("y".equals(userinput)){
            return true;
        }
        return false;
    }

    /**
     * Asks the player for a guess
     * Must not allow the player to crash the game
     * @returni int of the player's guess
     */
    private static int getGuess() {
        answer = (int) (Math.random()*(MAX_GUESS-MIN_GUESS)+MIN_GUESS);
        return -1;
    }

    /**
     * Gets a guess for the computer
     * Must not select a value that has already been guessed be either the
     * player or the computer
     * @return int of the computer's guess
     */
    private static int getCompGuess() {
        answer = (int) (Math.random()*(MAX_GUESS-MIN_GUESS)+MIN_GUESS);
        return -1;
    }
}