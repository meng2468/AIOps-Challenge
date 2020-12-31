VM_SERVER = 49.232.82.102
VM_PATH = /root/vm/

migrate_dir: 
	@scp -r * root@$(VM_SERVER):/$(VM_PATH)

pull_data:
	@scp -r root@$(VM_SERVER):/$(VM_PATH)/logs/* ../data