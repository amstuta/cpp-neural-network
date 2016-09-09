CC	=	g++

RM	=	rm -f

CPPFLAGS= -Wall -Wextra -Werror

NAME	=	nn_regression

SRCS	=	main.cpp \
				Network.cpp


OBJS	=	$(SRCS:.cpp=.o)

all:		$(NAME)

$(NAME):$(OBJS)
				$(CC) $(OBJS) -o $(NAME)

clean:
				$(RM) $(OBJS)

fclean:	clean
				$(RM) $(NAME)

re:			fclean all

.PHONY:	all clean fclean re
