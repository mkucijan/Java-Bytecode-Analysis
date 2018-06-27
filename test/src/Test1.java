public class Test1 {

	private static int parametar;

	public static void sayHello() {
		System.out.println("Hello.");
	}

    public static void main(String[] args) {
        int mylocal = 1;
        System.out.println("" + mylocal);
		if(mylocal>1) {
			mylocal = mylocal + 1;
		}
		while(mylocal<10) {
			System.out.println(".." + mylocal);
			mylocal = mylocal + 1;
		}
		for(int i=0; i<2; i++) {
			sayHello();
		}
    }
}