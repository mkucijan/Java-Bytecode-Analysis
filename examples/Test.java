public class Test {

	private static int parametar;

	public static void makeSwitch() {
		switch(parametar) { //line 6
			case 1 : System.out.println("1");
					 break;
			case 2 : System.out.println("2");
					 break;
			default: System.out.println("default");
					 break;
		}
		int inputValue=1;
		switch(inputValue) {
			case 1 : break;
			case 2 : break;
			case 3 : break;
			default: break;
		}
	}

	public static void sayHello() {
		System.out.println("Hello.");
	}

    public static void main(String[] args) {
        int mylocal = 1;
        System.out.println("" + mylocal);
		if(mylocal>1) {
			mylocal = mylocal + 1;
		} else {
			mylocal = 0;
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