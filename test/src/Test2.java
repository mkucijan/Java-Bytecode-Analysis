public class Test2 {

	private static int parametar;

	public static void sayHello() {
		System.out.println("Hello.");
	}

    public static void main(String[] args) {
        int mylocal = 1;
        System.out.println("" + mylocal);
		while(mylocal<10) {
			System.out.println(".." + mylocal);
            mylocal = mylocal + 1;
            for(int i=0; i<2; i++) {
                if(mylocal>1) {
                    mylocal = mylocal + 1;
                    sayHello();
                }
            }
		}
		
    }
}