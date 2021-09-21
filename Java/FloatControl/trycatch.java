public class trycatch {
    public static void main(String[] args) {
        int [] array = {1,2,3,4,5};
       

        try{
            System.out. println(array[10]);
        }catch (Exception e) {
            System.out.println("ERROR : " + e);
        }

        System.out.println("DONE!");
    }

}
