public class Main {
    public static void main(String[] args) {
        char[] letters = {'c','f','j'};
        char ans = nextGreatestLetter(letters, 'c');
        System.out.println(ans);
    }
    public static char nextGreatestLetter(char[] letters, char target) {
        int low = 0;
        int high = letters.length-1;
        while(low <= high){
            int mid = low + (high-low)/2;
            if(letters[mid] == target){
                return letters[low+1];
            }
            if(letters[mid] + 0 > target + 0){
                high = mid-1;
            }else {
               low = mid + 1;
            }
        }
    return letters[low];
    }
}
